import edgecase as ec


def main():
    # Create a social network
    print("=" * 70)
    print("Node Similarity Demo - Link Prediction in Social Networks")
    print("=" * 70)

    g = ec.Graph(directed=False)

    # Add people
    people = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
    for person in people:
        g.add_node(person)

    # Add friendships
    friendships = [
        ("Alice", "Bob"),
        ("Alice", "Carol"),
        ("Alice", "David"),
        ("Bob", "Carol"),
        ("Bob", "David"),
        ("Eve", "Frank"),
        ("Eve", "Grace"),
        ("Frank", "Grace"),
        ("Grace", "Henry"),
    ]

    for person1, person2 in friendships:
        g.add_edge(person1, person2)

    print(f"\nCreated social network with {g.count_nodes()} people and {g.count_edges()} friendships")
    print("\nFriendship groups:")
    print("  - Group 1: Alice, Bob, Carol, David (densely connected)")
    print("  - Group 2: Eve, Frank, Grace, Henry (another cluster)")

    # Example 1: Predict potential friendships within Group 1
    print("\n" + "=" * 70)
    print("Example 1: Finding Similar People (Potential New Friendships)")
    print("=" * 70)

    # Carol and David are not directly connected but have common friends
    people = ["Carol", "David"]

    print(f"\nComparing {people[0]} and {people[1]}:")
    print(f"  {people[0]}'s friends: {sorted(g.neighbors('Carol'))}")
    print(f"  {people[1]}'s friends: {sorted(g.neighbors('David'))}")

    print("\nSimilarity scores (higher = more similar):")

    for method in ["jaccard", "dice", "cosine", "invlogweighted"]:
        sim = g.pairwise_similarity(people, method=method)
        print(f"  {method:15s}: {sim[0][1]:.4f}")

    print(
        f"\nInterpretation: High similarity suggests {people[0]} and {people[1]} should be friends!"
    )

    # Example 2: Compare all metrics for multiple pairs
    print("\n" + "=" * 70)
    print("Example 2: Similarity Across Different Pairs")
    print("=" * 70)

    pairs_to_compare = [
        ("Alice", "Bob"),  # Already friends
        ("Carol", "David"),  # Same group, not friends
        ("Alice", "Eve"),  # Different groups
    ]

    print("\nComparing different relationship types:")
    print("                      Jaccard    Dice   Cosine  InvLogWgt")
    print("-" * 70)

    for person1, person2 in pairs_to_compare:
        scores = []
        for method in ["jaccard", "dice", "cosine", "invlogweighted"]:
            sim = g.pairwise_similarity([person1, person2], method=method)
            scores.append(sim[0][1])

        print(
            f"{person1:8s} - {person2:8s}: {scores[0]:7.4f}  {scores[1]:7.4f}  {scores[2]:7.4f}  {scores[3]:9.4f}"
        )

    # Example 3: Identify most similar person to each person
    print("\n" + "=" * 70)
    print("Example 3: Finding Most Similar Person for Each Individual")
    print("=" * 70)

    all_people = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace"]

    # Compute full similarity matrix
    sim_matrix = g.pairwise_similarity(all_people, method="jaccard")

    print("\nMost similar person (by Jaccard similarity):")
    for i, person in enumerate(all_people):
        # Find most similar (excluding self at diagonal)
        similarities = [(all_people[j], sim_matrix[i][j]) for j in range(len(all_people)) if i != j]
        most_similar = max(similarities, key=lambda x: x[1])

        print(f"  {person:8s} → {most_similar[0]:8s} (similarity: {most_similar[1]:.4f})")

    # Example 4: Link prediction - rank potential new friendships
    print("\n" + "=" * 70)
    print("Example 4: Link Prediction - Ranking Potential Friendships")
    print("=" * 70)

    print("\nTop 5 potential new friendships (using inverse log-weighted):")

    # Find all pairs that are NOT currently connected
    potential_links = []

    for i, person1 in enumerate(all_people):
        for person2 in all_people[i + 1 :]:
            # Check if they're not already friends
            if not g.has_edge(person1, person2):
                sim = g.pairwise_similarity([person1, person2], method="invlogweighted")
                potential_links.append((person1, person2, sim[0][1]))

    # Sort by similarity (higher = more likely to form friendship)
    potential_links.sort(key=lambda x: x[2], reverse=True)

    for rank, (p1, p2, score) in enumerate(potential_links[:5], 1):
        print(f"  {rank}. {p1:8s} - {p2:8s}  (score: {score:.4f})")

    # Example 5: Understanding metric differences
    print("\n" + "=" * 70)
    print("Example 5: Understanding Different Similarity Metrics")
    print("=" * 70)

    print("""
Metric Characteristics:

1. Jaccard Similarity
   - Most common, intersection over union
   - Range: [0, 1]
   - Good general-purpose metric

2. Dice Similarity
   - Gives more weight to common neighbors
   - Range: [0, 1]
   - Always ≥ Jaccard for same inputs

3. Cosine Similarity
   - Normalized dot product
   - Range: [0, 1]
   - Less affected by degree differences

4. Inverse Log-Weighted (Adamic-Adar)
   - Weights rare neighbors higher
   - Range: [0, ∞)
   - Best for link prediction tasks
   - Common high-degree neighbors contribute less
    """)

    # Example 6: Effect of common high-degree neighbors
    print("\n" + "=" * 70)
    print("Example 6: Effect of Hub Nodes on Similarity Metrics")
    print("=" * 70)

    # Create a graph with a hub node
    g2 = ec.Graph(directed=False)

    # Add people
    hub_people = ["Hub", "Person1", "Person2", "Person3", "Person4", "Person5"]
    for person in hub_people:
        g2.add_node(person)

    # Hub connects to many people
    hub_connections = [
        ("Hub", "Person1"),
        ("Hub", "Person2"),
        ("Hub", "Person3"),
        ("Hub", "Person4"),
        ("Hub", "Person5"),
    ]

    # Person1 and Person2 only connect through the hub
    for p1, p2 in hub_connections:
        g2.add_edge(p1, p2)

    print("\nTwo people connected only through a high-degree hub:")
    print("  Person1 ←→ Hub ←→ Person2")
    print(f"  Hub has {len(list(g2.neighbors('Hub')))} connections")

    # Compare metrics
    print("\nSimilarity between Person1 and Person2:")
    for method in ["jaccard", "dice", "cosine", "invlogweighted"]:
        sim = g2.pairwise_similarity(["Person1", "Person2"], method=method)
        print(f"  {method:15s}: {sim[0][1]:.4f}")

    print("\n→ Note: Inverse log-weighted gives lower score due to high-degree hub!")
    print("  This is useful because connections through hubs are less meaningful.")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
