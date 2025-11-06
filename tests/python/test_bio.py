"""Tests for bioinformatics functionality."""

import pytest
import polars as pl
from edgecase import protein_network, patient_similarity_network


# Protein network tests

def test_protein_network_basic() -> None:
    """Test creating a basic protein interaction network."""
    interactions = pl.DataFrame({
        "protein_a": ["P1", "P2", "P3"],
        "protein_b": ["P2", "P3", "P1"],
        "confidence": [0.9, 0.8, 0.7],
    })
    
    graph = protein_network(interactions, confidence_col="confidence")
    assert graph.num_nodes() == 3
    assert not graph.is_directed  # Protein interactions are undirected


def test_protein_network_confidence_filtering() -> None:
    """Test filtering by confidence score."""
    interactions = pl.DataFrame({
        "protein_a": ["P1", "P2", "P3"],
        "protein_b": ["P2", "P3", "P4"],
        "confidence": [0.9, 0.3, 0.8],
    })
    
    graph = protein_network(
        interactions,
        confidence_col="confidence",
        min_confidence=0.5,
    )
    
    # Should filter out P2-P3 interaction (confidence 0.3)
    assert graph.num_edges() == 2


# Patient similarity network tests

def test_patient_similarity_network_basic() -> None:
    """Test creating a patient similarity network."""
    patients = pl.DataFrame({
        "patient_id": ["P1", "P2", "P3"],
        "age": [45, 50, 48],
        "biomarker_1": [1.2, 1.5, 1.3],
        "biomarker_2": [0.8, 0.9, 0.85],
    })
    
    graph = patient_similarity_network(
        patients,
        features=["age", "biomarker_1", "biomarker_2"],
    )
    
    assert graph.num_nodes() == 3
    assert not graph.is_directed
    assert graph.is_weighted


@pytest.mark.slow
@pytest.mark.integration
def test_protein_network_real_data() -> None:
    """Integration test with larger protein network."""
    # This would test with actual STRING database data
    # Placeholder for now
    pass
