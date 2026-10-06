"""Exceptions raised by EdgeCase."""

from __future__ import annotations

__all__ = ["NodeNotFoundError", "EdgeNotFoundError"]


class NodeNotFoundError(ValueError, KeyError):
    """
    A node ID or label that doesn't exist in the graph.

    Raised whenever a method is given a node that isn't in the graph, whether as an integer ID
    or a string label. It is both a `ValueError` and a `KeyError`, so code written against
    either (older versions raised `KeyError` from some methods, such as `delete_node()` and
    `select_nodes_by_id()`, and `ValueError` from the rest) keeps working.

    Examples
    --------
    ```{python}
    import edgecase as ec

    graph = ec.Graph().add_nodes(["A", "B"])

    try:
        graph.get_node("Z")
    except ec.NodeNotFoundError as e:
        print(e)
    ```
    """

    # KeyError.__str__ would wrap the message in quotes
    __str__ = Exception.__str__


class EdgeNotFoundError(ValueError, KeyError):
    """
    An edge ID that doesn't exist in the graph.

    Raised whenever a method is given an edge ID that isn't in the graph. Like
    `NodeNotFoundError`, it is both a `ValueError` and a `KeyError`.

    Examples
    --------
    ```{python}
    import edgecase as ec

    graph = ec.Graph().add_nodes(["A", "B"]).add_edge("A", "B")

    try:
        graph.get_edge(99)
    except ec.EdgeNotFoundError as e:
        print(e)
    ```
    """

    __str__ = Exception.__str__
