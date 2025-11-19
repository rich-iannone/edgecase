import pytest
import edgecase as ec


class TestDegreeAggregationBasic:
    """Basic tests for degree aggregation methods."""

    def test_agg_degree_in_mean(self):
        """Test mean indegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        mean_in = G.get_agg_degree_in("mean")
        assert isinstance(mean_in, float)
        assert mean_in == 1.75  # 35 edges / 20 nodes

    def test_agg_degree_in_median(self):
        """Test median indegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        median_in = G.get_agg_degree_in("median")
        assert isinstance(median_in, float)
        assert median_in >= 0

    def test_agg_degree_in_sum(self):
        """Test sum indegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        sum_in = G.get_agg_degree_in("sum")
        assert sum_in == 35.0  # Total number of edges

    def test_agg_degree_in_min_max(self):
        """Test min and max indegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_star_graph(5)  # Hub has indegree 0, leaves have indegree 1

        min_in = G.get_agg_degree_in("min")
        max_in = G.get_agg_degree_in("max")

        assert min_in == 0.0
        assert max_in == 1.0

    def test_agg_degree_out_mean(self):
        """Test mean outdegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        mean_out = G.get_agg_degree_out("mean")
        assert isinstance(mean_out, float)
        assert mean_out == 1.75  # 35 edges / 20 nodes

    def test_agg_degree_out_sum(self):
        """Test sum outdegree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        sum_out = G.get_agg_degree_out("sum")
        assert sum_out == 35.0  # Total number of edges

    def test_agg_degree_total_mean(self):
        """Test mean total degree aggregation."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(20, 35, seed=23)

        mean_total = G.get_agg_degree_total("mean")
        assert isinstance(mean_total, float)
        assert mean_total == 3.5  # 70 degree sum / 20 nodes

    def test_agg_degree_total_undirected(self):
        """Test total degree aggregation works on undirected graphs."""
        G = ec.Graph(directed=False)
        G.add_cycle_graph(4)  # All nodes have degree 2

        mean_total = G.get_agg_degree_total("mean")
        assert mean_total == 2.0

        sum_total = G.get_agg_degree_total("sum")
        assert sum_total == 8.0  # 4 nodes * 2 degree each


class TestDegreeAggregationEdgeCases:
    """Edge case tests for degree aggregation."""

    def test_empty_graph(self):
        """Test aggregation on empty graph."""
        G = ec.Graph(directed=True)

        assert G.get_agg_degree_in("mean") == 0.0
        assert G.get_agg_degree_out("mean") == 0.0
        assert G.get_agg_degree_total("mean") == 0.0

    def test_single_node(self):
        """Test aggregation on single node graph."""
        G = ec.Graph(directed=True)
        G.add_node("A")

        assert G.get_agg_degree_in("mean") == 0.0
        assert G.get_agg_degree_out("mean") == 0.0
        assert G.get_agg_degree_total("mean") == 0.0
        assert G.get_agg_degree_in("sum") == 0.0
        assert G.get_agg_degree_in("min") == 0.0
        assert G.get_agg_degree_in("max") == 0.0

    def test_median_even_nodes(self):
        """Test median calculation with even number of nodes."""
        G = ec.Graph(directed=True)
        # Create graph where nodes have indegrees: 0, 1, 2, 3
        for node in ["A", "B", "C", "D"]:
            G.add_node(node)
        G.add_edge("A", "B")  # B has indegree 1
        G.add_edge("A", "C")  # C has indegree 1
        G.add_edge("B", "C")  # C has indegree 2
        G.add_edge("A", "D")  # D has indegree 1
        G.add_edge("B", "D")  # D has indegree 2
        G.add_edge("C", "D")  # D has indegree 3

        median = G.get_agg_degree_in("median")
        assert median == 1.5  # Median of [0, 1, 2, 3]

    def test_median_odd_nodes(self):
        """Test median calculation with odd number of nodes."""
        G = ec.Graph(directed=True)
        for node in ["A", "B", "C"]:
            G.add_node(node)
        G.add_edge("A", "B")  # B has indegree 1
        G.add_edge("A", "C")  # C has indegree 1

        median = G.get_agg_degree_in("median")
        assert median == 1.0  # Median of [0, 1, 1] sorted is [0, 1, 1], middle value is 1


class TestDegreeAggregationWithConditions:
    """Tests for conditional degree aggregation."""

    def test_conditions_filter_nodes(self):
        """Test that conditions properly filter nodes."""
        G = ec.Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            G.add_node(node)
        a_id, b_id = G.get_id_from_label("A"), G.get_id_from_label("B")
        c_id, d_id = G.get_id_from_label("C"), G.get_id_from_label("D")
        G.set_node_attrs("value", 10, nodes=[a_id, b_id])
        G.set_node_attrs("value", 5, nodes=[c_id, d_id])

        G.add_edge("A", "B", auto_create_nodes=True)
        G.add_edge("A", "C", auto_create_nodes=True)
        G.add_edge("B", "D", auto_create_nodes=True)

        # Only count nodes with value > 5
        mean_high_value = G.get_agg_degree_in(
            "mean", conditions=lambda n: n.attributes and n.attributes.get("value", 0) > 5
        )

        # Nodes A and B: A has indegree 0, B has indegree 1
        assert mean_high_value == 0.5

    def test_conditions_all_filtered(self):
        """Test when all nodes are filtered out."""
        G = ec.Graph(directed=True)
        G.add_gnm_graph(10, 15, seed=23)

        # Filter that matches no nodes
        result = G.get_agg_degree_in("mean", conditions=lambda n: False)
        assert result == 0.0


class TestDegreeAggregationErrors:
    """Test error handling in degree aggregation."""

    def test_invalid_agg_function(self):
        """Test error on invalid aggregation function."""
        G = ec.Graph(directed=True)
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="agg must be one of"):
            G.get_agg_degree_in("average")  # Invalid function name

    def test_undirected_graph_indegree(self):
        """Test error when using indegree on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="only works with directed graphs"):
            G.get_agg_degree_in("mean")

    def test_undirected_graph_outdegree(self):
        """Test error when using outdegree on undirected graph."""
        G = ec.Graph(directed=False)
        G.add_path_graph(3)

        with pytest.raises(ValueError, match="only works with directed graphs"):
            G.get_agg_degree_out("mean")


class TestDegreeAggregationRealWorld:
    """Real-world scenarios for degree aggregation."""

    def test_social_network_analysis(self):
        """Test degree aggregation on social network."""
        G = ec.Graph(directed=True)

        # Create a small social network
        for node in ["Alice", "Bob", "Charlie", "David", "Eve"]:
            G.add_node(node)

        # Alice is popular (many followers)
        G.add_edge("Bob", "Alice")
        G.add_edge("Charlie", "Alice")
        G.add_edge("David", "Alice")
        G.add_edge("Eve", "Alice")

        # Bob is active (follows many)
        G.add_edge("Bob", "Charlie")
        G.add_edge("Bob", "David")
        G.add_edge("Bob", "Eve")

        # Calculate follower statistics
        mean_followers = G.get_agg_degree_in("mean")
        max_followers = G.get_agg_degree_in("max")

        assert mean_followers == 1.4  # 7 edges / 5 nodes
        assert max_followers == 4.0  # Alice has 4 followers

        # Calculate following statistics
        mean_following = G.get_agg_degree_out("mean")
        max_following = G.get_agg_degree_out("max")

        assert mean_following == 1.4
        assert max_following == 4.0  # Bob follows 4 people

    def test_citation_network(self):
        """Test degree aggregation on citation network."""
        G = ec.Graph(directed=True)

        # Papers citing other papers
        for node in ["Paper1", "Paper2", "Paper3", "Paper4", "Paper5"]:
            G.add_node(node)

        # Paper1 is highly cited
        G.add_edge("Paper2", "Paper1")
        G.add_edge("Paper3", "Paper1")
        G.add_edge("Paper4", "Paper1")
        G.add_edge("Paper5", "Paper1")

        # Paper2 cites many papers
        G.add_edge("Paper2", "Paper3")
        G.add_edge("Paper2", "Paper4")

        # Average citations received
        avg_citations = G.get_agg_degree_in("mean")
        assert avg_citations == 1.2  # 6 edges / 5 nodes

        # Most cited paper
        max_citations = G.get_agg_degree_in("max")
        assert max_citations == 4.0  # Paper1

    def test_transportation_network(self):
        """Test degree aggregation on transportation network."""
        G = ec.Graph(directed=False)

        # Road network with intersections
        for node in ["A", "B", "C", "D", "E"]:
            G.add_node(node)
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("C", "D")
        G.add_edge("D", "E")
        G.add_edge("B", "D")  # Shortcut

        # Average connections per intersection
        avg_connections = G.get_agg_degree_total("mean")
        assert avg_connections == 2.0  # 10 total degree / 5 nodes

        # Most connected intersection
        max_connections = G.get_agg_degree_total("max")
        assert max_connections == 3.0  # B and D both have 3 connections


class TestDegreeAggregationAllFunctions:
    """Test all aggregation functions systematically."""

    def test_all_agg_functions_consistent(self):
        """Test that all aggregation functions work correctly."""
        G = ec.Graph(directed=True)
        G.add_path_graph(5)  # 1->2->3->4->5

        # Each node (except first) has indegree 1
        # Indegrees: [0, 1, 1, 1, 1]

        assert G.get_agg_degree_in("sum") == 4.0
        assert G.get_agg_degree_in("min") == 0.0
        assert G.get_agg_degree_in("max") == 1.0
        assert G.get_agg_degree_in("mean") == 0.8
        assert G.get_agg_degree_in("median") == 1.0

        # Each node (except last) has outdegree 1
        # Outdegrees: [1, 1, 1, 1, 0]

        assert G.get_agg_degree_out("sum") == 4.0
        assert G.get_agg_degree_out("min") == 0.0
        assert G.get_agg_degree_out("max") == 1.0
        assert G.get_agg_degree_out("mean") == 0.8
        assert G.get_agg_degree_out("median") == 1.0
