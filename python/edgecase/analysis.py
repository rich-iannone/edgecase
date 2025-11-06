"""High-level network analysis interface."""

from __future__ import annotations
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import polars as pl

if TYPE_CHECKING:
    from edgecase._core_rust import Graph

# Import algorithm functions - these get imported from the main module
# to ensure we get the correct versions (Rust or Python)
def _get_algorithms():
    """Lazily import algorithms to avoid circular imports."""
    import edgecase as ec
    return (
        ec.pagerank,
        ec.betweenness_centrality,
        ec.detect_communities
    )


class NetworkAnalyzer:
    """
    High-level interface for network analysis.
    
    Examples
    --------
    >>> import edgecase as ec
    >>> import polars as pl
    >>> 
    >>> # Load data
    >>> df = pl.read_csv("network_data.csv")
    >>> 
    >>> # Create analyzer
    >>> analyzer = ec.NetworkAnalyzer.from_polars(
    ...     df, source_col="from", target_col="to"
    ... )
    >>> 
    >>> # Run analysis
    >>> results = analyzer.analyze()
    >>> 
    >>> # Get summary table
    >>> summary = analyzer.summary_table()
    """
    
    def __init__(self, graph: Graph):
        """
        Initialize NetworkAnalyzer.
        
        Parameters
        ----------
        graph : Graph
            The graph to analyze
        """
        self.graph = graph
        self._centrality_cache: Dict[str, Any] = {}
        self._community_cache: Optional[Dict[str, int]] = None
    
    @classmethod
    def from_polars(
        cls,
        df: pl.DataFrame,
        source_col: str,
        target_col: str,
        **kwargs: Any,
    ) -> "NetworkAnalyzer":
        """
        Create analyzer from Polars DataFrame.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input data
        source_col : str
            Source node column
        target_col : str
            Target node column
        **kwargs
            Additional arguments passed to from_polars
            
        Returns
        -------
        NetworkAnalyzer
        """
        from edgecase.polars_integration import from_polars
        graph = from_polars(df, source_col, target_col, **kwargs)
        return cls(graph)
    
    def compute_centrality(
        self,
        measures: Optional[List[str]] = None,
        force_recompute: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute centrality measures.
        
        Parameters
        ----------
        measures : list of str, optional
            Which measures to compute. Options: 'pagerank', 'betweenness'
        force_recompute : bool, default False
            Whether to force recomputation even if cached
            
        Returns
        -------
        dict
            Dictionary mapping measure names to node centrality scores
        """
        if measures is None:
            measures = ['pagerank', 'betweenness']
        
        # Get algorithm functions
        pagerank_fn, betweenness_fn, _ = _get_algorithms()
        
        results = {}
        
        for measure in measures:
            if measure in self._centrality_cache and not force_recompute:
                results[measure] = self._centrality_cache[measure]
            elif measure == 'pagerank':
                results[measure] = pagerank_fn(self.graph)
                self._centrality_cache[measure] = results[measure]
            elif measure == 'betweenness' and betweenness_fn is not None:
                results[measure] = betweenness_fn(self.graph)
                self._centrality_cache[measure] = results[measure]
        
        return results
    
    def find_communities(self, force_recompute: bool = False) -> Dict[str, int]:
        """
        Detect communities in the network.
        
        Parameters
        ----------
        force_recompute : bool, default False
            Whether to force recomputation
            
        Returns
        -------
        dict
            Mapping of node IDs to community IDs
        """
        if self._community_cache is None or force_recompute:
            _, _, detect_communities_fn = _get_algorithms()
            if detect_communities_fn is not None:
                self._community_cache = detect_communities_fn(self.graph)
            else:
                self._community_cache = {}
        return self._community_cache
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run complete network analysis.
        
        Returns
        -------
        dict
            Dictionary containing all analysis results
        """
        return {
            'basic_stats': {
                'n_nodes': self.graph.num_nodes(),
                'n_edges': self.graph.num_edges(),
                'directed': self.graph.is_directed,
            },
            'centrality': self.compute_centrality(),
            'communities': self.find_communities(),
        }
    
    def summary_table(self) -> pl.DataFrame:
        """
        Generate a summary table of network properties.
        
        Returns
        -------
        pl.DataFrame
            Summary statistics as a DataFrame
        """
        from edgecase.tables import tbl_network_summary
        stats = tbl_network_summary(self.graph)
        return pl.DataFrame([stats])
