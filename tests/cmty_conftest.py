"""Shared configuration for community detection tests."""

import edgecase as ec


class GraphWithAutoCreate(ec.Graph):
    """Graph subclass that auto-creates nodes in add_edge by default."""

    def add_edge(
        self, source, target=None, weight=None, edge_id=None, auto_create_nodes=True, **attributes
    ):
        """Override add_edge to default auto_create_nodes=True."""
        return super().add_edge(source, target, weight, edge_id, auto_create_nodes, **attributes)
