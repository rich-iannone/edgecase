import json

import edgecase as ec
import polars as pl
import pytest


class TestDeterministicGenerators:
    """Test deterministic graph generators."""

    def test_add_path_graph(self):
        """Generate a path graph."""
        G = ec.Graph()
        G.add_path_graph(5)

        assert G.count_nodes() == 5
        assert G.count_edges() == 4

    def test_add_cycle_graph(self):
        """Generate a cycle graph."""
        G = ec.Graph()
        G.add_cycle_graph(4)

        assert G.count_nodes() == 4
        assert G.count_edges() == 4

    def test_add_star_graph(self):
        """Generate a star graph."""
        G = ec.Graph()
        G.add_star_graph(5)

        assert G.count_nodes() == 5
        assert G.count_edges() == 4

    def test_add_full_graph(self):
        """Generate a complete graph."""
        G = ec.Graph()
        G.add_full_graph(4)

        assert G.count_nodes() == 4
        assert G.count_edges() == 6  # n*(n-1)/2 for undirected

    def test_add_full_graph_directed(self):
        """Generate a complete directed graph."""
        G = ec.Graph(directed=True)
        G.add_full_graph(4)

        assert G.count_nodes() == 4
        assert G.count_edges() == 12  # n*(n-1) for directed

    def test_add_balanced_tree(self):
        """Generate a balanced tree."""
        G = ec.Graph(directed=True)
        G.add_balanced_tree(k=2, h=3)

        assert G.count_nodes() == 15  # 1 + 2 + 4 + 8
        assert G.count_edges() == 14  # n - 1 for tree

    def test_add_grid_2d(self):
        """Generate a 2D grid."""
        G = ec.Graph()
        G.add_grid_2d(3, 4)

        assert G.count_nodes() == 12
        # Grid edges: (rows-1)*cols + rows*(cols-1)
        assert G.count_edges() == 2 * 4 + 3 * 3

    def test_add_prism_graph(self):
        """Generate a prism graph."""
        G = ec.Graph()
        G.add_prism_graph(5)

        assert G.count_nodes() == 10  # Two cycles of 5
        assert G.count_edges() == 15  # 5 + 5 + 5


class TestRandomGenerators:
    """Test random graph generators."""

    def test_add_gnp_graph_basic(self):
        """Generate Erdős-Rényi G(n,p) graph."""
        G = ec.Graph()
        G.add_gnp_graph(50, 0.1, seed=23)

        assert G.count_nodes() == 50
        # With p=0.1, expect roughly 0.1 * 50*49/2 ≈ 122 edges
        assert 80 < G.count_edges() < 160

    def test_add_gnp_graph_reproducibility(self):
        """G(n,p) with same seed produces same graph."""
        G1 = ec.Graph()
        G1.add_gnp_graph(30, 0.2, seed=123)

        G2 = ec.Graph()
        G2.add_gnp_graph(30, 0.2, seed=123)

        assert G1.count_edges() == G2.count_edges()

    def test_add_gnp_graph_probability_bounds(self):
        """G(n,p) respects probability bounds."""
        # p=0 should create no edges
        G_empty = ec.Graph()
        G_empty.add_gnp_graph(20, 0.0, seed=23)
        assert G_empty.count_edges() == 0

        # p=1 should create complete graph
        G_complete = ec.Graph()
        G_complete.add_gnp_graph(10, 1.0, seed=23)
        assert G_complete.count_edges() == 45  # 10*9/2

    def test_add_gnm_graph_basic(self):
        """Generate Erdős-Rényi G(n,m) graph."""
        G = ec.Graph()
        G.add_gnm_graph(50, 100, seed=23)

        assert G.count_nodes() == 50
        assert G.count_edges() == 100

    def test_add_gnm_graph_directed(self):
        """G(n,m) works with directed graphs."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 50, directed=True, seed=23)

        assert G.count_nodes() == 20
        assert G.count_edges() == 50

    def test_add_gnm_graph_max_edges(self):
        """G(n,m) rejects m > max possible edges."""
        G = ec.Graph()

        with pytest.raises(ValueError, match="Cannot create"):
            G.add_gnm_graph(10, 100, seed=23)  # max is 45

    @pytest.mark.parametrize("directed", [False, True])
    def test_add_gnm_graph_large_is_sampled(self, directed):
        # Past about a million possible edges the pairs are sampled rather than listed, which
        # needed 3 GB at 20,000 nodes; the graph is still simple, exact and seeded
        def make(seed):
            return ec.Graph(directed=directed).add_gnm_graph(n=20_000, m=40_000, seed=seed)

        G = make(1)
        assert (G.count_nodes(), G.count_edges()) == (20_000, 40_000)
        assert G.is_simple()
        assert G.get_edges_as_tuples() == make(1).get_edges_as_tuples()
        assert G.get_edges_as_tuples() != make(2).get_edges_as_tuples()

    def test_add_gnm_graph_large_and_dense(self):
        # Nearly every pair: the sampler picks the few pairs to leave out instead
        n = 1_450  # 1,050,525 possible edges, just past the listing threshold
        G = ec.Graph().add_gnm_graph(n=n, m=n * (n - 1) // 2 - 10, seed=1)
        assert G.count_edges() == n * (n - 1) // 2 - 10
        assert G.is_simple()

    def test_add_pa_graph_basic(self):
        """Generate Barabási-Albert graph."""
        G = ec.Graph()
        G.add_pa_graph(100, m=3, seed=23)

        assert G.count_nodes() == 100
        # Initial clique of m+1 nodes has m+1 choose 2 edges
        # Each additional node adds m edges
        # Total: (m+1)*m/2 + (n - m - 1) * m
        expected_edges = 4 * 3 // 2 + (100 - 4) * 3
        assert G.count_edges() == expected_edges

    def test_add_pa_graph_parameters(self):
        """PA graph validates parameters."""
        G = ec.Graph()

        with pytest.raises(ValueError, match="m must be at least 1"):
            G.add_pa_graph(50, m=0)

        with pytest.raises(ValueError, match="m must be less than n"):
            G.add_pa_graph(10, m=10)

    def test_add_smallworld_graph_basic(self):
        """Generate Watts-Strogatz small-world graph."""
        G = ec.Graph()
        G.add_smallworld_graph(100, k=6, p=0.1, seed=23)

        assert G.count_nodes() == 100
        # Should have roughly n*k/2 edges (some may be rewired)
        assert 280 < G.count_edges() <= 300

    def test_add_smallworld_graph_no_rewiring(self):
        """Small-world with p=0 creates ring lattice."""
        G = ec.Graph()
        G.add_smallworld_graph(20, k=4, p=0.0, seed=23)

        assert G.count_nodes() == 20
        assert G.count_edges() == 40  # 20 * 4 / 2

    def test_add_smallworld_graph_parameters(self):
        """Small-world validates parameters."""
        G = ec.Graph()

        with pytest.raises(ValueError, match="k must be even"):
            G.add_smallworld_graph(100, k=5, p=0.1)

        with pytest.raises(ValueError, match="k must be less than n"):
            G.add_smallworld_graph(10, k=20, p=0.1)

    @pytest.mark.parametrize("seed", range(40))
    def test_smallworld_has_no_duplicate_edges(self, seed):
        """Rewiring never creates an edge that already exists (or is still to be visited)."""
        g = ec.Graph()
        g.add_smallworld_graph(n=8, k=4, p=0.2, seed=seed)
        pairs = [tuple(sorted((s, t))) for _, s, t in g.get_edges_as_tuples()]
        assert len(pairs) == len(set(pairs)) == 16


class TestMethodChaining:
    """Test that generators support method chaining."""

    def test_chain_multiple_generators(self):
        """Chain multiple graph generators."""
        G = ec.Graph().add_path_graph(5).add_cycle_graph(4).add_star_graph(3)

        assert G.count_nodes() == 12

    def test_chain_with_selection(self):
        """Chain generators with selection operations."""
        G = (
            ec.Graph()
            .add_full_graph(4)
            .select_nodes_by_id([1, 2, 3, 4])
            .set_node_attrs_ws(group="A")
        )

        assert G.count_nodes() == 4
        # get_node() takes a label, so use '1' string label
        assert G.get_node("1").attributes["group"] == "A"

    def test_chain_random_and_deterministic(self):
        """Chain random and deterministic generators."""
        G = ec.Graph().add_path_graph(10).add_gnp_graph(20, 0.1, seed=23)

        assert G.count_nodes() == 30


class TestGraphProperties:
    """Test properties of generated graphs."""

    def test_path_graph_connectivity(self):
        """Path graph has correct connectivity."""
        G = ec.Graph().add_path_graph(5)

        # First and last nodes have degree 1
        # Middle nodes have degree 2
        degrees = [
            G.get_degree("1"),
            G.get_degree("2"),
            G.get_degree("3"),
            G.get_degree("4"),
            G.get_degree("5"),
        ]
        assert degrees == [1, 2, 2, 2, 1]

    def test_star_graph_hub(self):
        """Star graph has one hub node."""
        G = ec.Graph().add_star_graph(6)

        # First node should be the hub with degree 5
        assert G.get_degree("1") == 5

        # Other nodes should have degree 1
        for i in range(2, 7):
            assert G.get_degree(str(i)) == 1

    def test_cycle_graph_uniform_degree(self):
        """Cycle graph has uniform degree 2."""
        G = ec.Graph().add_cycle_graph(5)

        for i in range(1, 6):
            assert G.get_degree(str(i)) == 2

    def test_full_graph_uniform_degree(self):
        """Complete graph has uniform degree n-1."""
        G = ec.Graph().add_full_graph(5)

        for i in range(1, 6):
            assert G.get_degree(str(i)) == 4


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimum_sizes(self):
        """Test minimum valid sizes."""
        G1 = ec.Graph().add_path_graph(1)
        assert G1.count_nodes() == 1 and G1.count_edges() == 0

        G2 = ec.Graph().add_cycle_graph(3)
        assert G2.count_nodes() == 3 and G2.count_edges() == 3

        G3 = ec.Graph().add_star_graph(2)
        assert G3.count_nodes() == 2 and G3.count_edges() == 1

    def test_empty_gnp_graph(self):
        """G(n,p) with p=0 creates isolated nodes."""
        G = ec.Graph().add_gnp_graph(10, 0.0)

        assert G.count_nodes() == 10
        assert G.count_edges() == 0

    def test_gnm_zero_edges(self):
        """G(n,m) with m=0 creates isolated nodes."""
        G = ec.Graph().add_gnm_graph(10, 0)

        assert G.count_nodes() == 10
        assert G.count_edges() == 0


class TestPerformance:
    """Test performance with larger graphs."""

    def test_large_path(self):
        """Generate large path graph."""
        G = ec.Graph().add_path_graph(1000)

        assert G.count_nodes() == 1000
        assert G.count_edges() == 999

    def test_large_gnp(self):
        """Generate large random graph."""
        G = ec.Graph().add_gnp_graph(500, 0.01, seed=23)

        assert G.count_nodes() == 500
        # Should complete quickly

    def test_large_pa(self):
        """Generate large scale-free graph."""
        G = ec.Graph().add_pa_graph(1000, m=3, seed=23)

        assert G.count_nodes() == 1000


class TestRandomGraphDirectedOverride:
    """Regression: a directed= that differed from the graph's raised AttributeError after the
    nodes had already been added."""

    @pytest.mark.parametrize("method,args", [("add_gnp_graph", {"p": 1.0}), ("add_gnm_graph", {"m": 12})])
    def test_directed_on_undirected_graph(self, method, args):
        # Both orientations of each of the 6 pairs: 12 edges, every pair doubled
        g = ec.Graph().add_nodes(["X"])
        getattr(g, method)(n=4, directed=True, seed=1, **args)
        assert not g.is_directed()
        assert g.count_nodes() == 5
        assert g.count_edges() == 12
        assert g.count_multi_edges() == 6
        x = g.get_id_from_label("X")
        assert all(x not in (s, t) for _, s, t in g.get_edges_as_tuples())

    @pytest.mark.parametrize("method,args", [("add_gnp_graph", {"p": 1.0}), ("add_gnm_graph", {"m": 6})])
    def test_undirected_on_directed_graph(self, method, args):
        # Each of the 6 pairs once, in one direction
        g = ec.Graph(directed=True)
        getattr(g, method)(n=4, directed=False, seed=1, **args)
        assert g.is_directed()
        assert g.count_nodes() == 4
        assert g.count_edges() == 6
        assert g.count_mutual_node_pairs() == 0

    def test_invalid_args_add_nothing(self):
        g = ec.Graph()
        with pytest.raises(ValueError):
            g.add_gnp_graph(n=4, p=2.0, directed=True)
        assert g.count_nodes() == 0


class TestDocstringScenarios:
    """Values quoted in the generator docstrings."""

    def test_path_and_star(self):
        path = ec.Graph().add_path_graph(n=5)
        assert path.count_edges() == 4
        assert path.get_node_df()["degree"].to_list() == [1, 2, 2, 2, 1]
        assert path.diameter() == 4

        line = ec.Graph(directed=True).add_path_graph(n=4, type="stop", rel="next")
        assert line.get_edges_str_repr(use_labels=True) == ["1->2", "2->3", "3->4"]

        star = ec.Graph().add_star_graph(n=7)
        assert star.diameter() == 2
        assert ec.Graph().add_path_graph(n=7).diameter() == 6

        gateway = ec.Graph().add_node(label="Gateway").add_star_graph(n=4).add_edge("Gateway", "2")
        assert gateway.get_edges_str_repr(use_labels=True) == [
            "2--3",
            "2--4",
            "2--5",
            "Gateway--2",
        ]

    def test_cycles_and_full_graphs(self):
        ring = ec.Graph(directed=True).add_cycle_graph(n=4, rel="next")
        assert ring.get_edges_str_repr(use_labels=True)[-1] == "4->1"
        assert ec.Graph().add_cycle_graph(n=6).is_bipartite()
        assert not ec.Graph().add_cycle_graph(n=7).is_bipartite()

        assert ec.Graph().add_full_graph(n=4).count_edges() == 6
        assert ec.Graph(directed=True).add_full_graph(n=4).count_edges() == 12
        assert ec.Graph().add_full_graph(n=8).count_edges() == 28
        looped = ec.Graph().add_full_graph(n=3, keep_loops=True)
        assert "1--1" in looped.get_edges_str_repr(use_labels=True)
        assert not looped.is_simple()

    def test_balanced_trees(self):
        tree = ec.Graph(directed=True).add_balanced_tree(k=2, h=3)
        assert tree.count_nodes() == 15
        assert tree.get_edges_str_repr(use_labels=True)[:4] == [
            "1->2",
            "1->3",
            "2->4",
            "2->5",
        ]
        chain = ec.Graph().add_balanced_tree(k=1, h=3)
        assert chain.get_edges_str_repr(use_labels=True) == ["1--2", "2--3", "3--4"]
        assert ec.Graph().add_balanced_tree(k=3, h=2).count_nodes() == 13

    def test_grids_and_ladders(self):
        grid = ec.Graph().add_grid_2d(rows=3, cols=4)
        assert grid.count_edges() == 17
        assert sorted(grid.get_node_df()["degree"].to_list()) == [2] * 4 + [3] * 6 + [4] * 2

        square = ec.Graph().add_grid_2d(rows=4, cols=4)
        long = ec.Graph().add_grid_2d(rows=2, cols=8)
        assert (square.count_edges(), long.count_edges()) == (24, 22)
        assert (square.diameter(), long.diameter()) == (6, 8)

        ladder = ec.Graph().add_ladder_graph(n=4)
        assert ladder.count_edges() == 10
        assert ladder.is_isomorphic(ec.Graph().add_grid_2d(rows=2, cols=4))
        prism = ec.Graph().add_prism_graph(n=6)
        assert prism.count_edges() - ec.Graph().add_ladder_graph(n=6).count_edges() == 2

    def test_prisms(self):
        prism = ec.Graph().add_prism_graph(n=5)
        assert (prism.count_nodes(), prism.count_edges()) == (10, 15)
        assert prism.is_k_regular(3)
        cube = ec.Graph().add_prism_graph(n=4)
        assert cube.get_node_df()["label"].to_list()[:5] == [
            "1_top",
            "2_top",
            "3_top",
            "4_top",
            "5_bottom",
        ]
        assert cube.is_isomorphic(ec.Graph().add_hypercube_graph(3))

    def test_multipartite_graphs(self):
        k23 = ec.Graph().add_complete_bipartite_graph(n1=2, n2=3)
        assert k23.count_edges() == 6
        assert k23.get_node_df()["degree"].to_list() == [3, 3, 2, 2, 2]
        assert k23.is_planar()
        assert not ec.Graph().add_complete_bipartite_graph(n1=3, n2=3).is_planar()

        groups = ec.Graph().add_complete_multipartite_graph(sizes=[1, 2, 3])
        assert groups.count_edges() == 11
        assert groups.get_node_df()["degree"].to_list() == [5, 4, 4, 3, 3, 3]
        octahedron = ec.Graph().add_complete_multipartite_graph(sizes=[2, 2, 2])
        assert octahedron.is_isomorphic(ec.Graph().add_named_graph("octahedral"))

    def test_turan_graphs(self):
        graph = ec.Graph().add_turan_graph(n=7, r=3)
        counts = graph.get_node_df()["part"].value_counts().sort("part")
        assert counts["count"].to_list() == [2, 2, 3]
        assert (graph.count_edges(), graph.clique_number()) == (16, 3)

        two = ec.Graph().add_turan_graph(n=8, r=2)
        four = ec.Graph().add_turan_graph(n=8, r=4)
        assert (two.count_edges(), four.count_edges()) == (16, 24)
        assert two.clique_number() == 2
        assert four.clique_number() == 4

    def test_wheels_and_circulants(self):
        wheel = ec.Graph().add_wheel_graph(n=6)
        assert wheel.count_edges() == 10
        assert wheel.get_node_df()["degree"].to_list() == [5, 3, 3, 3, 3, 3]
        assert ec.Graph().add_wheel_graph(n=4).is_isomorphic(ec.Graph().add_full_graph(n=4))

        graph = ec.Graph().add_circulant_graph(n=8, offsets=[1, 3])
        assert graph.is_k_regular(4)
        assert graph.count_edges() == 16
        arrows = ec.Graph(directed=True).add_circulant_graph(n=4, offsets=[1])
        assert arrows.get_edges_str_repr(use_labels=True) == ["1->2", "2->3", "3->4", "4->1"]
        wagner = ec.Graph().add_circulant_graph(n=8, offsets=[1, 4])
        assert wagner.count_edges() == 12
        assert wagner.is_k_regular(3)

    def test_barbells_lollipops_and_tadpoles(self):
        graph = ec.Graph().add_barbell_graph(m1=4, m2=2)
        assert (graph.count_nodes(), graph.count_edges()) == (10, 15)
        scores = graph.betweenness_centrality(return_dict=True)
        rounded = {x: round(scores[graph.get_id_from_label(x)], 2) for x in "34567"}
        assert rounded == {"3": 0.0, "4": 0.5, "5": 0.56, "6": 0.56, "7": 0.5}
        bridge = ec.Graph().add_barbell_graph(m1=4, m2=0)
        assert len(bridge.articulation_points()) == 2
        assert len(graph.articulation_points()) == 4
        assert len(graph.bridges()) == 3

        lollipop = ec.Graph().add_lollipop_graph(m=4, n=3)
        assert (lollipop.count_nodes(), lollipop.count_edges()) == (7, 9)
        assert lollipop.diameter() == 4
        assert (lollipop.get_degree("4"), lollipop.get_degree("7")) == (4, 1)
        assert ec.Graph().add_lollipop_graph(m=5, n=1).diameter() == 2
        assert ec.Graph().add_lollipop_graph(m=5, n=3).diameter() == 4

        tadpole = ec.Graph().add_tadpole_graph(m=5, n=2)
        assert (tadpole.count_nodes(), tadpole.count_edges()) == (7, 7)
        assert (tadpole.get_degree("5"), tadpole.girth()) == (3, 5)
        square = ec.Graph().add_tadpole_graph(m=4, n=3)
        assert (square.count_edges(), square.girth()) == (7, 4)
        assert lollipop.girth() == 3

    def test_windmills(self):
        graph = ec.Graph().add_windmill_graph(n=3, k=3)
        assert (graph.count_nodes(), graph.count_edges()) == (7, 9)
        hub = graph.articulation_points()
        assert [graph.get_label_from_id(n) for n in hub] == ["1"]
        rest = graph.subgraph([n for n in graph.get_node_ids() if n not in hub])
        assert rest.count_weakly_connected_components() == 3
        fours = ec.Graph().add_windmill_graph(n=3, k=4)
        assert (fours.count_nodes(), fours.count_edges(), fours.get_degree("1")) == (10, 18, 9)
        assert graph.get_degree("1") == 6

    def test_hypercubes_and_grids(self):
        cube = ec.Graph().add_hypercube_graph(d=3)
        assert (cube.count_nodes(), cube.count_edges(), cube.diameter()) == (8, 12, 3)
        neighbors = cube.get_neighbors("000")
        assert sorted(cube.get_label_from_id(n) for n in neighbors) == ["001", "010", "100"]
        assert cube.shortest_path_length("001", "110") == 3
        tesseract = ec.Graph().add_hypercube_graph(d=4)
        assert (tesseract.count_nodes(), tesseract.count_edges()) == (16, 32)

        grid = ec.Graph().add_grid_graph(dims=[2, 3, 4])
        assert (grid.count_nodes(), grid.count_edges()) == (24, 46)
        assert len(grid.get_neighbors("(1,1,1)")) == 5
        torus = ec.Graph().add_grid_graph(dims=[4, 4], periodic=True)
        assert torus.count_edges() == 32
        assert torus.is_k_regular(4)
        flat = ec.Graph().add_grid_graph(dims=[3, 6])
        tube = ec.Graph().add_grid_graph(dims=[3, 6], periodic=[False, True])
        assert (flat.count_edges(), tube.count_edges()) == (27, 30)
        assert (flat.diameter(), tube.diameter()) == (7, 5)

    def test_lattices(self):
        small = ec.Graph().add_triangular_lattice_graph(rows=2, cols=4)
        assert (small.count_nodes(), small.count_edges()) == (9, 16)
        assert len(small.list_triangles()) == 8
        lattice = ec.Graph().add_triangular_lattice_graph(rows=4, cols=8)
        assert len(lattice.get_neighbors("(2,2)")) == 6
        square = ec.Graph().add_grid_2d(rows=5, cols=5)
        assert lattice.count_nodes() == square.count_nodes() == 25
        assert (lattice.count_edges(), square.count_edges()) == (56, 40)
        assert (lattice.diameter(), square.diameter()) == (6, 8)

        row = ec.Graph().add_hexagonal_lattice_graph(rows=1, cols=2)
        assert (row.count_nodes(), row.count_edges(), row.girth()) == (10, 11, 6)
        honeycomb = ec.Graph().add_hexagonal_lattice_graph(rows=3, cols=3)
        assert len(honeycomb.list_triangles()) == 0
        assert honeycomb.get_node_df()["degree"].max() == 3
        assert lattice.get_node_df()["degree"].max() == 6

    def test_kneser_and_johnson_graphs(self):
        petersen = ec.Graph().add_named_graph("petersen")
        kneser = ec.Graph().add_kneser_graph(n=5, k=2)
        assert (kneser.count_nodes(), kneser.count_edges()) == (10, 15)
        assert kneser.is_isomorphic(petersen)
        neighbors = kneser.get_neighbors("0,1")
        assert sorted(kneser.get_label_from_id(n) for n in neighbors) == ["2,3", "2,4", "3,4"]
        johnson = ec.Graph().add_johnson_graph(n=5, k=2)
        assert johnson.is_isomorphic(kneser.complement())
        assert johnson.count_edges() == 30
        assert johnson.is_k_regular(6)

        small = ec.Graph().add_johnson_graph(n=4, k=2)
        assert (small.count_nodes(), small.count_edges()) == (6, 12)
        assert small.is_k_regular(4)
        neighbors = small.get_neighbors("0,1")
        assert sorted(small.get_label_from_id(n) for n in neighbors) == [
            "0,2",
            "0,3",
            "1,2",
            "1,3",
        ]
        assert small.is_isomorphic(ec.Graph().add_named_graph("octahedral"))

    def test_mycielski_graphs(self):
        graph = ec.Graph().add_mycielski_graph(n=4)
        assert (graph.count_nodes(), graph.count_edges()) == (11, 20)
        assert (len(graph.list_triangles()), graph.clique_number()) == (0, 2)
        cycle = ec.Graph().add_mycielski_graph(n=3)
        assert cycle.is_isomorphic(ec.Graph().add_cycle_graph(n=5))
        assert len(set(cycle.greedy_color().values())) == 3
        assert len(set(graph.greedy_color().values())) == 4
        # The shell drawing puts node i + 5 inside node i, so it must be i's shadow
        cycle_nodes = {graph.get_id_from_label(str(i)) for i in range(1, 6)}
        for i in range(1, 6):
            node = graph.get_id_from_label(str(i))
            shadow = graph.get_id_from_label(str(i + 5))
            on_cycle = set(graph.get_neighbors(node)) & cycle_nodes
            assert set(graph.get_neighbors(shadow)) & cycle_nodes == on_cycle

    def test_trees(self):
        tree = ec.Graph(directed=True).add_binomial_tree(order=3)
        assert (tree.count_nodes(), tree.get_out_degree("1")) == (8, 3)
        depths = tree.single_source_shortest_path_length("1", return_dict=True)
        assert [sum(1 for d in depths.values() if d == level) for level in range(4)] == [1, 3, 3, 1]
        bigger = ec.Graph(directed=True).add_binomial_tree(order=4)
        children = bigger.get_successors("1")
        assert sorted(bigger.get_label_from_id(n) for n in children) == ["2", "3", "5", "9"]
        copy = [bigger.get_id_from_label(str(i)) for i in range(9, 17)]
        assert bigger.subgraph(copy).is_isomorphic(tree)

        heap = ec.Graph(directed=True).add_full_rary_tree(r=2, n=6)
        assert heap.get_edges_str_repr(use_labels=True) == [
            "1->2",
            "1->3",
            "2->4",
            "2->5",
            "3->6",
        ]
        assert sorted(heap.get_label_from_id(n) for n in heap.get_successors("2")) == ["4", "5"]
        binary = ec.Graph().add_full_rary_tree(r=2, n=13)
        ternary = ec.Graph().add_full_rary_tree(r=3, n=13)
        assert (binary.diameter(), ternary.diameter()) == (6, 4)

    def test_lcf_and_petersen_families(self):
        heawood = ec.Graph().add_lcf_graph(n=14, shifts=[5, -5], repeats=7)
        assert (heawood.count_edges(), heawood.girth()) == (21, 6)
        assert heawood.is_isomorphic(ec.Graph().add_named_graph("heawood"))
        cube = ec.Graph().add_lcf_graph(n=8, shifts=[3, -3], repeats=4)
        assert cube.is_isomorphic(ec.Graph().add_hypercube_graph(d=3))

        petersen = ec.Graph().add_generalized_petersen_graph(5, 2)
        assert (petersen.count_nodes(), petersen.count_edges()) == (10, 15)
        assert petersen.is_isomorphic(ec.Graph().add_named_graph("petersen"))
        girths = {k: ec.Graph().add_generalized_petersen_graph(10, k).girth() for k in (1, 2, 3)}
        assert girths == {1: 4, 2: 5, 3: 6}
        # Spokes join outer node i to inner node i + 10
        desargues = ec.Graph().add_generalized_petersen_graph(10, 3)
        assert all(desargues.has_edge(str(i), str(i + 10)) for i in range(1, 11))

    def test_named_graphs(self):
        petersen = ec.Graph().add_named_graph("petersen")
        assert (petersen.count_nodes(), petersen.count_edges(), petersen.girth()) == (10, 15, 5)
        tutte = ec.Graph().add_named_graph("tutte")
        assert tutte.is_k_regular(3)
        assert tutte.is_planar()
        assert tutte.node_connectivity() == 3

        kite = ec.Graph().add_named_graph("Krackhardt kite")
        kite.set_node_attrs("betweenness", kite.betweenness_centrality(return_dict=True))
        nodes = kite.get_node_df()
        assert nodes.sort("degree", descending=True)["label"][0] == "4"
        assert nodes.sort("betweenness", descending=True)["label"][0] == "8"

    def test_gnp_and_gnm_graphs(self):
        graph = ec.Graph().add_gnp_graph(n=50, p=0.1, seed=23)
        assert (graph.count_nodes(), graph.count_edges()) == (50, 109)
        assert round(graph.density(), 3) == 0.089
        assert ec.Graph(directed=True).add_gnp_graph(n=50, p=0.1, seed=23).count_edges() == 230
        connected = [
            ec.Graph().add_gnp_graph(n=100, p=p, seed=23).is_connected() for p in (0.02, 0.05, 0.1)
        ]
        assert connected == [False, True, True]
        sparse = ec.Graph().add_gnp_graph(n=30, p=0.05, seed=23)
        dense = ec.Graph().add_gnp_graph(n=30, p=0.15, seed=23)
        assert sparse.count_weakly_connected_components() == 8
        assert dense.is_connected()

        assert ec.Graph().add_gnm_graph(n=50, m=100, seed=23).count_edges() == 100
        assert ec.Graph(directed=True).add_gnm_graph(n=4, m=12).count_edges() == 12
        first = ec.Graph().add_gnm_graph(n=8, m=10, seed=1)
        second = ec.Graph().add_gnm_graph(n=8, m=10, seed=5)
        assert first.get_node_df()["degree"].to_list() == [4, 3, 3, 2, 2, 3, 1, 2]
        assert second.get_node_df()["degree"].to_list() == [3, 2, 2, 3, 3, 2, 2, 3]

    def test_preferential_attachment(self):
        graph = ec.Graph().add_pa_graph(n=100, m=3, seed=23)
        assert graph.count_edges() == 294
        degrees = graph.get_node_df()["degree"].sort(descending=True).to_list()
        assert degrees[:5] == [30, 29, 23, 18, 13]
        assert min(degrees) == 3
        tree = ec.Graph(directed=True).add_pa_graph(n=6, m=1, seed=23)
        assert tree.get_edges_str_repr() == ["1->2", "3->2", "4->3", "5->3", "6->2"]

        pa = ec.Graph().add_pa_graph(n=50, m=2, seed=23)
        uniform = ec.Graph().add_gnm_graph(n=50, m=pa.count_edges(), seed=23)
        assert pa.count_edges() == 97

        def top(g):
            return g.get_node_df()["degree"].sort(descending=True).to_list()[:5]

        assert top(pa) == [22, 13, 12, 11, 9]
        assert top(uniform) == [8, 7, 7, 7, 7]

    def test_small_worlds(self):
        assert ec.Graph().add_smallworld_graph(n=100, k=6, p=0.1, seed=23).count_edges() == 300
        results = []
        for p in (0.0, 0.1, 1.0):
            g = ec.Graph().add_smallworld_graph(n=100, k=6, p=p, seed=23)
            results.append((round(g.average_clustering_coefficient(), 3), g.diameter()))
        assert results == [(0.6, 17), (0.473, 8), (0.052, 5)]
        ring = ec.Graph().add_smallworld_graph(n=30, k=4, p=0.0)
        small = ec.Graph().add_smallworld_graph(n=30, k=4, p=0.1, seed=23)
        assert (ring.diameter(), small.diameter()) == (8, 5)

    def test_random_geometric_graphs(self):
        graph = ec.Graph().add_random_geometric_graph(n=100, radius=0.2, seed=23)
        assert graph.count_edges() == 527
        assert graph.get_label_from_id(1) == "[0.541348, 0.246466]"
        spatial = ec.Graph().add_random_geometric_graph(n=50, radius=0.3, dim=3, seed=123)
        assert spatial.count_edges() == 88
        assert len(json.loads(spatial.get_label_from_id(1))) == 3
        manhattan = ec.Graph().add_random_geometric_graph(n=100, radius=0.2, p_norm=1.0, seed=23)
        assert manhattan.count_edges() == 360
        near = ec.Graph().add_random_geometric_graph(n=60, radius=0.1, seed=23)
        far = ec.Graph().add_random_geometric_graph(n=60, radius=0.2, seed=23)
        assert near.count_weakly_connected_components() == 24
        assert sorted(len(c) for c in far.weakly_connected_components()) == [3, 57]

    def test_configuration_models(self):
        graph = ec.Graph().configuration_model([3, 2, 2, 2, 1], seed=2)
        assert dict(graph.get_node_df().select("label", "degree").rows()) == {
            "0": 3,
            "1": 2,
            "2": 2,
            "3": 2,
            "4": 1,
        }
        assert graph.get_edges_str_repr(use_labels=True) == ["1--3", "0--1", "3--4", "0--2", "0--2"]
        assert not graph.is_simple()
        assert graph.simplify().count_edges() == 4

        directed = ec.Graph().configuration_model([2, 1, 1, 0], directed=True, seed=4)
        assert [
            (directed.get_in_degree(n), directed.get_out_degree(n))
            for n in sorted(directed.get_node_ids())
        ] == [(2, 2), (1, 1), (1, 1), (0, 0)]

        observed = ec.Graph()
        for _ in range(4):
            observed.add_full_graph(n=4)
        null = ec.Graph().configuration_model(observed.get_node_df()["degree"].to_list(), seed=23)
        assert observed.average_clustering_coefficient() == 1.0
        assert null.count_edges() == 24 and null.is_k_regular(3) and null.is_simple()
        assert null.average_clustering_coefficient() == 0.0

    def test_degree_preserving_rewire(self):
        graph = (
            ec.Graph()
            .add_nodes(["A", "B", "C", "D", "E", "F"])
            .add_edges(
                [("A", "B"), ("B", "C"), ("D", "E"), ("E", "F"), ("A", "D"), ("B", "E"), ("C", "F")]
            )
        )
        once = graph.degree_preserving_rewire(n_swaps=1, seed=3)
        assert once.get_edges_str_repr(use_labels=True) == [
            "A--E",
            "B--C",
            "B--D",
            "E--F",
            "A--D",
            "B--E",
            "C--F",
        ]
        rewired = graph.degree_preserving_rewire(seed=3)
        assert rewired.get_edges_str_repr(use_labels=True) == [
            "A--E",
            "B--F",
            "B--D",
            "C--D",
            "A--F",
            "B--E",
            "C--E",
        ]

        def degrees(g):
            return g.get_node_df().select("label", "degree").rows()

        assert degrees(graph) == degrees(rewired)

        observed = ec.Graph()
        for _ in range(10):
            observed.add_full_graph(n=6)
        observed.add_edges([(str(c * 6 + 1), str((c + 1) % 10 * 6 + 2)) for c in range(10)])
        null = observed.degree_preserving_rewire(seed=23)
        assert round(observed.average_clustering_coefficient(), 3) == 0.889
        assert round(null.average_clustering_coefficient(), 3) == 0.034
        assert null.is_simple()

    def test_expected_degree_graphs(self):
        weights = [10] * 5 + [2] * 45
        graph = ec.Graph().expected_degree_graph(weights, seed=23)
        assert graph.count_edges() == 70
        degree = graph.get_node_df()["degree"]
        assert degree[:5].to_list() == [11, 14, 10, 10, 7]
        assert round(degree[5:].mean(), 2) == 1.96

        even = ec.Graph().expected_degree_graph([3] * 20, seed=1)
        heavy = ec.Graph().expected_degree_graph([12] * 2 + [2] * 18, seed=1)
        assert even.get_node_df()["degree"].max() == 6
        top = heavy.get_node_df().sort("degree", descending=True)
        assert top["degree"][:3].to_list() == [12, 8, 4]
        assert sorted(top["label"][:2].to_list()) == ["0", "1"]

    def test_stochastic_block_models(self):
        def within(g):
            return sum((s <= 10) == (t <= 10) for _, s, t in g.get_edges_as_tuples())

        graph = ec.Graph().generate_stochastic_block_model(
            sizes=[10, 10], p_matrix=[0.8, 0.05, 0.05, 0.8], seed=23
        )
        assert (graph.count_edges(), within(graph)) == (82, 76)
        communities = graph.cmty_louvain(return_dict=True)
        by_label = {int(graph.get_label_from_id(n)): c for n, c in communities.items()}
        assert [by_label[i] for i in range(20)] == [0] * 10 + [1] * 10

        blurred = ec.Graph().generate_stochastic_block_model(
            sizes=[10, 10], p_matrix=[0.5, 0.3, 0.3, 0.5], seed=23
        )
        assert (blurred.count_edges(), within(blurred)) == (82, 48)
        cliques = ec.Graph().generate_stochastic_block_model(
            sizes=[3, 2], p_matrix=[1.0, 0.0, 0.0, 1.0]
        )
        assert cliques.get_edges_str_repr(use_labels=True) == ["0--1", "0--2", "1--2", "3--4"]

    def test_caveman_graphs(self):
        graph = ec.Graph().add_caveman_graph(num_cliques=3, clique_size=4)
        assert (graph.count_edges(), graph.count_weakly_connected_components()) == (18, 3)
        assert graph.modularity("community") == pytest.approx(2 / 3)
        connected = ec.Graph().add_connected_caveman_graph(num_cliques=3, clique_size=4)
        assert (connected.count_edges(), connected.is_connected()) == (18, True)

        caves = ec.Graph().add_connected_caveman_graph(num_cliques=4, clique_size=5)
        degrees = caves.get_node_df()["degree"].value_counts().sort("degree")
        assert degrees.rows() == [(3, 4), (4, 12), (5, 4)]
        assert caves.partition_quality("community")[0] == pytest.approx(0.9)
        assert round(caves.modularity("community"), 2) == 0.65
        assert ec.Graph().add_ring_of_cliques(num_cliques=4, clique_size=5).count_edges() == 44

    def test_ring_of_cliques(self):
        graph = ec.Graph().add_ring_of_cliques(num_cliques=6, clique_size=3)
        assert (graph.count_nodes(), graph.count_edges()) == (18, 24)
        assert ec.Graph().add_ring_of_cliques(num_cliques=6, clique_size=5).count_edges() == 66

        ring = ec.Graph().add_ring_of_cliques(num_cliques=30, clique_size=3)
        pairs = {n: c // 2 for n, c in ring.get_node_attrs("community").items()}
        assert round(ring.modularity("community"), 3) == 0.717
        assert round(ring.modularity(pairs), 3) == 0.808

    def test_relaxed_caveman(self):
        def relaxed(p):
            return ec.Graph().add_relaxed_caveman_graph(num_cliques=5, clique_size=6, p=p, seed=7)

        assert relaxed(0.2).count_edges() == 75
        coverage = [relaxed(p).partition_quality("community")[0] for p in (0.0, 0.1, 0.3, 0.6)]
        assert [round(c, 2) for c in coverage] == [1.0, 0.92, 0.75, 0.53]
        assert [round(relaxed(p).modularity("community"), 2) for p in (0.1, 0.6)] == [0.72, 0.33]

    def test_planted_and_random_partitions(self):
        graph = ec.Graph().add_planted_partition_graph(
            num_groups=4, group_size=25, p_in=0.3, p_out=0.03, seed=1
        )
        assert graph.count_edges() == 477
        assert round(graph.partition_quality("community")[0], 2) == 0.77
        found = graph.cmty_louvain(return_dict=True)
        assert len(set(found.values())) == 4
        assert round(graph.modularity(found), 2) == round(graph.modularity("community"), 2) == 0.51

        def coverage(p_out):
            g = ec.Graph().add_planted_partition_graph(
                num_groups=4, group_size=10, p_in=0.6, p_out=p_out, seed=1
            )
            return round(g.partition_quality("community")[0], 2)

        assert (coverage(0.02), coverage(0.15)) == (0.89, 0.55)

        groups = ec.Graph().add_random_partition_graph(
            sizes=[10, 20, 40], p_in=0.4, p_out=0.02, seed=3
        )
        # unrounded: 9.25 is a tie, which older polars rounds up and newer polars to even
        means = groups.get_node_df().group_by("community").agg(pl.col("degree").mean())
        assert means.sort("community")["degree"].to_list() == pytest.approx([4.7, 9.25, 16.4])
        unequal = ec.Graph().add_random_partition_graph(
            sizes=[6, 12, 24], p_in=0.4, p_out=0.02, seed=3
        )
        equal = ec.Graph().add_random_partition_graph(
            sizes=[14, 14, 14], p_in=0.4, p_out=0.02, seed=3
        )
        assert (unequal.count_edges(), equal.count_edges()) == (165, 135)

    def test_gaussian_random_partition(self):
        def group_sizes(**kwargs):
            g = ec.Graph().add_gaussian_random_partition_graph(**kwargs)
            counts = g.get_node_df()["community"].value_counts()
            return counts.sort("community")["count"].to_list()

        sizes = group_sizes(n=200, mean_size=20, shape=2, p_in=0.3, p_out=0.01, seed=5)
        assert sizes == [20, 24, 19, 17, 22, 14, 19, 23, 16, 20, 6]
        small = dict(n=60, mean_size=10, p_in=0.5, p_out=0.02, seed=2)
        assert group_sizes(shape=2, **small) == [6, 12, 9, 15, 10, 8]
        assert group_sizes(shape=50, **small) == [9, 10, 10, 11, 10, 10]

    def test_lfr_benchmarks(self):
        graph = ec.Graph().add_lfr_graph(n=500, mu=0.2, average_degree=10, seed=4)
        assert graph.count_edges() == 2507
        assert graph.get_node_df()["community"].n_unique() == 43
        degrees = graph.get_node_df()["degree"]
        assert (degrees.min(), round(degrees.mean(), 1), degrees.max()) == (3, 10.0, 25)
        found = graph.cmty_louvain(return_dict=True)
        assert round(graph.partition_quality("community")[0], 2) == 0.8
        assert round(graph.modularity("community"), 2) == 0.76
        assert round(graph.modularity(found), 2) == 0.77

        def lfr(mu):
            return ec.Graph().add_lfr_graph(
                n=100, mu=mu, average_degree=8, min_community=20, max_community=40, seed=3
            )

        clear, mixed = lfr(0.1), lfr(0.5)
        for g in (clear, mixed):
            counts = g.get_node_df()["community"].value_counts().sort("community")
            assert counts["count"].to_list() == [22, 34, 24, 20]
        assert round(clear.modularity("community"), 2) == 0.65
        assert round(mixed.modularity("community"), 2) == 0.25
