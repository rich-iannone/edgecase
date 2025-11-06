"""
Fraud Detection Example: Transaction Network Analysis

This example demonstrates how to detect suspicious patterns in financial
transaction networks, including money laundering rings and fraud communities.
"""

import polars as pl
import edgecase as ec
from datetime import datetime, timedelta

# Example 1: Basic Transaction Network
print("=" * 60)
print("Example 1: Transaction Network Analysis")
print("=" * 60)

# Sample transaction data
transactions = pl.DataFrame({
    "from_account": ["A001", "A002", "A003", "A004", "A001", "A002"],
    "to_account": ["A002", "A003", "A004", "A001", "A005", "A006"],
    "amount": [50000, 45000, 40000, 38000, 500, 1200],
    "timestamp": [
        "2024-01-01 10:00:00",
        "2024-01-01 10:05:00",
        "2024-01-01 10:10:00",
        "2024-01-01 10:15:00",
        "2024-01-01 11:00:00",
        "2024-01-01 11:30:00",
    ],
    "transaction_type": ["wire", "wire", "wire", "wire", "card", "card"],
})

print("\nTransaction Data:")
print(transactions)

# Create transaction network (directed graph)
G = ec.transaction_network(
    transactions,
    min_amount=10000,  # Filter small transactions
)

print(f"\nNetwork Statistics:")
print(f"  Accounts: {G.num_nodes()}")
print(f"  Transactions: {G.num_edges()}")
print(f"  Directed: {G.is_directed}")

# Example 2: Detect Suspicious Circular Flows
print("\n" + "=" * 60)
print("Example 2: Money Laundering Detection (Circular Flows)")
print("=" * 60)

suspicious = ec.find_suspicious_patterns(
    G,
    min_cycle_amount=10000,
    min_community_size=3,
)

print(f"\nSuspicious Patterns Found:")
print(f"  Circular flows (cycles): {len(suspicious['cycles'])}")
print(f"  Potential fraud rings: {len(suspicious['fraud_rings'])}")

if suspicious['cycles']:
    print("\n⚠️  ALERT: Circular transaction pattern detected!")
    print("   This could indicate money laundering or structuring.")
    for i, cycle in enumerate(suspicious['cycles'][:3], 1):
        print(f"\n   Cycle {i}: {' → '.join(cycle)}")

# Example 3: High-Velocity Account Detection
print("\n" + "=" * 60)
print("Example 3: High-Velocity Account Analysis")
print("=" * 60)

print("""
High-velocity indicators (typical fraud patterns):

1. Many transactions in short time window
2. Round-number amounts (e.g., $10,000, $50,000)
3. Just-below-reporting thresholds ($9,999)
4. Rapid sequence of transfers
5. Unusual time-of-day patterns
""")

# Example transaction velocity analysis
velocity_df = transactions.with_columns([
    pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
]).group_by("from_account").agg([
    pl.count().alias("transaction_count"),
    pl.col("amount").sum().alias("total_amount"),
    (pl.col("timestamp").max() - pl.col("timestamp").min()).alias("time_span"),
])

print("\nAccount Velocity Metrics:")
print(velocity_df)

# Example 4: Network-Based Risk Scoring
print("\n" + "=" * 60)
print("Example 4: Network-Based Risk Scoring")
print("=" * 60)

print("""
Risk scoring using graph metrics:

1. Degree Centrality:
   - High out-degree → distributing funds (potential hub)
   - High in-degree → receiving funds (money mule?)
   
2. Betweenness Centrality:
   - High betweenness → intermediary in transaction chains
   - Could be layering in money laundering scheme
   
3. Community Detection:
   - Tightly connected groups → organized fraud rings
   - Isolated clusters → separate criminal networks
   
4. Temporal Patterns:
   - Burst activity → structured transactions
   - Consistent patterns → legitimate business
""")

# Analyze network structure
analyzer = ec.NetworkAnalyzer(G)
centrality = analyzer.compute_centrality(measures=['betweenness'])

print("\nHigh-Risk Accounts (by betweenness centrality):")
for account, score in sorted(
    centrality['betweenness'].items(),
    key=lambda x: x[1],
    reverse=True
)[:5]:
    print(f"  {account}: {score:.4f} (intermediary risk)")

# Example 5: Real-World Patterns
print("\n" + "=" * 60)
print("Example 5: Common Fraud Patterns to Detect")
print("=" * 60)

print("""
1. Smurfing/Structuring:
   Pattern: Many small transactions just under reporting threshold
   Graph: Star pattern with one center, many edges
   Detection: High out-degree + amounts near threshold

2. Layering:
   Pattern: Complex chain of transfers to obscure origin
   Graph: Long paths through multiple intermediaries
   Detection: High betweenness centrality + long shortest paths

3. Integration:
   Pattern: Illicit funds mixed with legitimate transactions
   Graph: Connection between suspicious and normal clusters
   Detection: Bridge nodes between communities

4. Money Mule Networks:
   Pattern: Accounts rapidly receiving and forwarding funds
   Graph: High turnover nodes in transaction chains
   Detection: High velocity + short-lived accounts

5. Trade-Based Laundering:
   Pattern: Over/under-invoicing in international trade
   Graph: Bi-directional flows with mismatched values
   Detection: Asymmetric edge weights in cycles
""")

# Example 6: Automated Alert System
print("\n" + "=" * 60)
print("Example 6: Automated Alert Framework")
print("=" * 60)

print("""
Production fraud detection pipeline:

1. Streaming ingestion:
   - Real-time transaction events
   - Incremental graph updates
   
2. Pattern detection:
   - Cycle detection for circular flows
   - Velocity analysis for burst activity
   - Community detection for fraud rings
   
3. Risk scoring:
   - Combine multiple graph metrics
   - Machine learning on graph features
   - Anomaly detection
   
4. Alert generation:
   - Priority based on risk score
   - Context: related accounts, patterns
   - Recommended actions
   
5. Feedback loop:
   - Analyst reviews
   - False positive reduction
   - Model retraining
""")

print("\n" + "=" * 60)
print("For more examples, see:")
print("  - examples/bioinformatics.py")
print("  - docs/fraud_detection_guide.md")
print("=" * 60)
