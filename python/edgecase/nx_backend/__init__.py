"""edgecase as a NetworkX backend.

NetworkX dispatches to edgecase when asked to, for example with
`nx.pagerank(G, backend="edgecase")` or `nx.config.backend_priority.algos = ["edgecase"]`,
converting its graphs with `from_networkx()` (see `edgecase.nx_backend.graph`). The
functions edgecase implements are in `edgecase.nx_backend.algorithms`.
"""

from edgecase.nx_backend.graph import EdgecaseGraph, from_networkx, to_networkx

__all__ = ["EdgecaseGraph", "from_networkx", "to_networkx"]
