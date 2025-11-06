"""Tests for fraud detection functionality."""

import pytest
import polars as pl
from edgecase import transaction_network, find_suspicious_patterns


# Transaction network tests

def test_transaction_network_basic() -> None:
    """Test creating a basic transaction network."""
    transactions = pl.DataFrame({
        "from_account": ["A1", "A2", "A3"],
        "to_account": ["A2", "A3", "A1"],
        "amount": [1000.0, 2000.0, 1500.0],
    })
    
    graph = transaction_network(transactions)
    assert graph.num_nodes() == 3
    assert graph.is_directed  # Transactions are directional


def test_transaction_network_amount_filtering() -> None:
    """Test filtering by transaction amount."""
    transactions = pl.DataFrame({
        "from_account": ["A1", "A2", "A3"],
        "to_account": ["A2", "A3", "A4"],
        "amount": [50000.0, 500.0, 40000.0],
    })
    
    graph = transaction_network(transactions, min_amount=10000.0)
    
    # Should filter out 500.0 transaction
    assert graph.num_edges() == 2


# Suspicious pattern tests

def test_find_suspicious_patterns_empty_graph() -> None:
    """Test pattern detection on empty graph."""
    from edgecase import Graph
    graph = Graph(directed=True)
    
    patterns = find_suspicious_patterns(graph)
    assert len(patterns['cycles']) == 0


def test_find_cycles_in_transaction_network() -> None:
    """Test detecting circular transaction patterns."""
    transactions = pl.DataFrame({
        "from_account": ["A1", "A2", "A3"],
        "to_account": ["A2", "A3", "A1"],
        "amount": [10000.0, 20000.0, 15000.0],
    })
    
    graph = transaction_network(transactions)
    patterns = find_suspicious_patterns(graph, min_cycle_amount=5000.0)
    
    # Should detect the cycle A1 -> A2 -> A3 -> A1
    assert 'cycles' in patterns


@pytest.mark.benchmark
def test_transaction_network_performance(benchmark) -> None:
    """Benchmark transaction network creation with large dataset."""
    n = 100000
    transactions = pl.DataFrame({
        "from_account": [f"A{i}" for i in range(n)],
        "to_account": [f"A{(i+1) % n}" for i in range(n)],
        "amount": [1000.0 + i for i in range(n)],
    })
    
    def create_network() -> None:
        transaction_network(transactions)
    
    benchmark(create_network)
