"""
Bioinformatics Example: Protein-Protein Interaction Network Analysis

This example demonstrates how to analyze protein interaction networks
from the STRING database for pathway analysis and drug target identification.
"""

import polars as pl
import edgecase as ec

# Example 1: Basic Protein Network
print("=" * 60)
print("Example 1: Protein-Protein Interaction Network")
print("=" * 60)

# Sample data (normally from STRING database)
interactions = pl.DataFrame({
    "protein_a": ["TP53", "MDM2", "ATM", "BRCA1", "CHEK2"],
    "protein_b": ["MDM2", "TP53", "BRCA1", "TP53", "BRCA1"],
    "confidence": [0.999, 0.999, 0.900, 0.850, 0.920],
    "experiment": [True, True, True, False, True],
})

print("\nProtein Interactions:")
print(interactions)

# Create network with high confidence threshold
G = ec.protein_network(
    interactions,
    confidence_col="confidence",
    min_confidence=0.85,
)

print(f"\nNetwork Statistics:")
print(f"  Proteins (nodes): {G.num_nodes()}")
print(f"  Interactions (edges): {G.num_edges()}")
print(f"  Directed: {G.is_directed}")

# Example 2: Pathway Enrichment Analysis
print("\n" + "=" * 60)
print("Example 2: Finding Key Proteins (Hub Analysis)")
print("=" * 60)

# Analyze network
analyzer = ec.NetworkAnalyzer(G)

# Get centrality measures
centrality = analyzer.compute_centrality(measures=['pagerank', 'betweenness'])

print("\nProtein Importance (PageRank):")
for protein, score in sorted(
    centrality['pagerank'].items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {protein}: {score:.4f}")

# Example 3: Patient Similarity Network for Clinical Trials
print("\n" + "=" * 60)
print("Example 3: Patient Similarity Network")
print("=" * 60)

# Sample clinical trial data
patients = pl.DataFrame({
    "patient_id": ["P001", "P002", "P003", "P004", "P005"],
    "age": [45, 52, 48, 51, 46],
    "tumor_size": [2.5, 3.2, 2.8, 3.0, 2.6],
    "biomarker_her2": [1.2, 2.8, 1.5, 2.5, 1.3],
    "biomarker_er": [0.8, 0.2, 0.7, 0.3, 0.9],
    "treatment_arm": ["A", "B", "A", "B", "A"],
})

print("\nPatient Data:")
print(patients)

# Create patient similarity network
patient_net = ec.patient_similarity_network(
    patients,
    features=["age", "tumor_size", "biomarker_her2", "biomarker_er"],
    similarity_threshold=0.7,
)

print(f"\nPatient Network:")
print(f"  Patients: {patient_net.num_nodes()}")
print(f"  Similarities: {patient_net.num_edges()}")

# Identify patient subgroups
communities = ec.detect_communities(patient_net)
print(f"\nPatient Subgroups: {len(set(communities.values()))}")

# Example 4: Drug Target Identification
print("\n" + "=" * 60)
print("Example 4: Drug Target Identification Strategy")
print("=" * 60)

print("""
Workflow for identifying promising drug targets:

1. Build protein interaction network from STRING/BioGRID
2. Calculate centrality measures:
   - Betweenness: proteins bridging modules
   - PageRank: globally important proteins
   - Degree: highly connected hubs
3. Identify disease-associated proteins
4. Find druggable targets in network neighborhoods
5. Validate with pathway enrichment analysis

Key metrics for drug targets:
- High betweenness → disrupting protein may affect multiple pathways
- High degree → hub protein, potential master regulator
- Community membership → functional module identification
""")

print("\nNext Steps:")
print("  1. Load real data from STRING database")
print("  2. Integrate gene expression data")
print("  3. Perform pathway enrichment")
print("  4. Generate publication-ready tables with Great Tables")

# Example 5: Network Summary Table (for publications)
print("\n" + "=" * 60)
print("Example 5: Publication-Ready Summary")
print("=" * 60)

summary = ec.tbl_network_summary(G, include_centrality=True)
print("\nNetwork Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("For more examples, see:")
print("  - examples/fraud_detection.py")
print("  - docs/bioinformatics_guide.md")
print("=" * 60)
