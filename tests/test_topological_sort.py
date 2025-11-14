import pytest
from edgecase import Graph


class TestTopologicalSort:
    """Test topological_sort() method."""

    def test_simple_dag(self):
        """Test basic topological sort on simple DAG."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")

        order = g.topological_sort()  # Returns list[int] (node IDs)

        # Convert IDs to labels for checking
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # Check all nodes present
        assert set(order_labels) == {"A", "B", "C", "D"}

        # Verify edge constraints
        assert order_labels.index("A") < order_labels.index("B")
        assert order_labels.index("A") < order_labels.index("C")
        assert order_labels.index("B") < order_labels.index("D")
        assert order_labels.index("C") < order_labels.index("D")

    def test_linear_chain(self):
        """Test topological sort on linear chain."""
        g = Graph(directed=True)
        nodes = ["1", "2", "3", "4", "5"]
        for node in nodes:
            g.add_node(label=node)
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1])

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]
        assert order_labels == nodes  # Should be in exact order

    def test_empty_graph(self):
        """Test topological sort on empty graph."""
        g = Graph(directed=True)
        order = g.topological_sort()
        assert order == []

    def test_single_node(self):
        """Test topological sort with single node."""
        g = Graph(directed=True)
        g.add_node(label="A")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]
        assert order_labels == ["A"]

    def test_disconnected_components(self):
        """Test topological sort on disconnected DAG."""
        g = Graph(directed=True)
        # Component 1: A -> B
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        # Component 2: C -> D
        g.add_node(label="C")
        g.add_node(label="D")
        g.add_edge("C", "D")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # All nodes should be present
        assert set(order_labels) == {"A", "B", "C", "D"}

        # Constraints within components
        assert order_labels.index("A") < order_labels.index("B")
        assert order_labels.index("C") < order_labels.index("D")

    def test_cycle_detection(self):
        """Test that cycles are detected."""
        g = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g.add_node(label=node)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")  # Creates cycle

        with pytest.raises(ValueError, match="cycle.*not a DAG"):
            g.topological_sort()

    def test_self_loop(self):
        """Test that self-loops are detected as cycles."""
        g = Graph(directed=True)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")
        g.add_edge("B", "B")  # Self-loop

        with pytest.raises(ValueError, match="cycle"):
            g.topological_sort()

    def test_undirected_graph_error(self):
        """Test that undirected graphs raise error."""
        g = Graph(directed=False)
        g.add_node(label="A")
        g.add_node(label="B")
        g.add_edge("A", "B")

        with pytest.raises(ValueError, match="directed"):
            g.topological_sort()

    def test_complex_dag(self):
        """Test more complex DAG with multiple valid orderings."""
        g = Graph(directed=True)
        nodes = ["A", "B", "C", "D", "E", "F"]
        for node in nodes:
            g.add_node(label=node)

        # Build a DAG with multiple paths
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E"), ("C", "F"), ("E", "F")]
        for src, tgt in edges:
            g.add_edge(src, tgt)

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # Verify all constraints
        for src, tgt in edges:
            assert order_labels.index(src) < order_labels.index(tgt)

    def test_diamond_graph(self):
        """Test diamond-shaped DAG."""
        g = Graph(directed=True)
        for node in ["top", "left", "right", "bottom"]:
            g.add_node(label=node)

        g.add_edge("top", "left")
        g.add_edge("top", "right")
        g.add_edge("left", "bottom")
        g.add_edge("right", "bottom")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        assert order_labels[0] == "top"
        assert order_labels[-1] == "bottom"
        assert {"left", "right"} == set(order_labels[1:3])

    def test_task_scheduling(self):
        """Test realistic task scheduling scenario."""
        g = Graph(directed=True)

        tasks = {
            "wake_up": [],
            "shower": ["wake_up"],
            "breakfast": ["wake_up"],
            "dress": ["shower"],
            "leave": ["breakfast", "dress"],
        }

        for task in tasks:
            g.add_node(label=task)

        for task, deps in tasks.items():
            for dep in deps:
                g.add_edge(dep, task)

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # Verify dependencies
        for task, deps in tasks.items():
            task_idx = order_labels.index(task)
            for dep in deps:
                dep_idx = order_labels.index(dep)
                assert dep_idx < task_idx, f"{dep} should come before {task}"

    def test_multiple_roots(self):
        """Test DAG with multiple source nodes."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(label=node)

        # Multiple roots (A, B) pointing to common node C
        g.add_edge("A", "C")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("C", "E")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # A and B should come before C
        assert order_labels.index("A") < order_labels.index("C")
        assert order_labels.index("B") < order_labels.index("C")

        # C should come before D and E
        assert order_labels.index("C") < order_labels.index("D")
        assert order_labels.index("C") < order_labels.index("E")

    def test_multiple_sinks(self):
        """Test DAG with multiple sink nodes."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D"]:
            g.add_node(label=node)

        # Single root pointing to multiple sinks
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("A", "D")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        assert order_labels[0] == "A"
        assert set(order_labels[1:]) == {"B", "C", "D"}

    def test_build_system(self):
        """Test realistic build system dependencies."""
        g = Graph(directed=True)

        files = ["config.h", "types.h", "utils.h", "utils.c", "main.c"]

        for f in files:
            g.add_node(label=f)

        # Dependencies
        g.add_edge("config.h", "types.h")
        g.add_edge("types.h", "utils.h")
        g.add_edge("utils.h", "utils.c")
        g.add_edge("utils.h", "main.c")
        g.add_edge("config.h", "main.c")

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # config.h must be first
        assert order_labels[0] == "config.h"

        # Verify all dependencies
        deps = [
            ("config.h", "types.h"),
            ("types.h", "utils.h"),
            ("utils.h", "utils.c"),
            ("utils.h", "main.c"),
            ("config.h", "main.c"),
        ]

        for src, tgt in deps:
            assert order_labels.index(src) < order_labels.index(tgt)

    def test_university_courses(self):
        """Test course prerequisite graph."""
        g = Graph(directed=True)

        courses = {
            "CS101": [],
            "CS102": ["CS101"],
            "MATH101": [],
            "CS201": ["CS102", "MATH101"],
            "CS301": ["CS201"],
        }

        for course in courses:
            g.add_node(label=course)

        for course, prereqs in courses.items():
            for prereq in prereqs:
                g.add_edge(prereq, course)

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # Verify prerequisites
        for course, prereqs in courses.items():
            course_idx = order_labels.index(course)
            for prereq in prereqs:
                prereq_idx = order_labels.index(prereq)
                assert prereq_idx < course_idx


class TestTopologicalSortIntegration:
    """Integration tests for topological_sort()."""

    def test_with_is_dag(self):
        """Test topological_sort works only on DAGs."""
        # DAG should work
        g1 = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g1.add_node(label=node)
        g1.add_edge("A", "B")
        g1.add_edge("B", "C")

        assert g1.is_dag()
        order = g1.topological_sort()
        assert len(order) == 3

        # Cyclic should fail
        g2 = Graph(directed=True)
        for node in ["A", "B", "C"]:
            g2.add_node(label=node)
        g2.add_edge("A", "B")
        g2.add_edge("B", "C")
        g2.add_edge("C", "A")

        assert not g2.is_dag()
        with pytest.raises(ValueError):
            g2.topological_sort()

    def test_isolated_nodes(self):
        """Test DAG with isolated nodes."""
        g = Graph(directed=True)
        for node in ["A", "B", "C", "D", "E"]:
            g.add_node(label=node)

        # Only connect A -> B
        g.add_edge("A", "B")
        # C, D, E are isolated

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        # All nodes should be in result
        assert set(order_labels) == {"A", "B", "C", "D", "E"}

        # A must come before B
        assert order_labels.index("A") < order_labels.index("B")

    def test_large_chain(self):
        """Test topological sort on large chain."""
        g = Graph(directed=True)
        n = 100

        for i in range(n):
            g.add_node(label=str(i))

        for i in range(n - 1):
            g.add_edge(str(i), str(i + 1))

        order = g.topological_sort()  # Returns list[int]
        order_labels = [g.get_label_from_id(nid) for nid in order]

        assert len(order_labels) == n
        # Should be in exact sequential order
        assert order_labels == [str(i) for i in range(n)]
