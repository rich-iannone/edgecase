use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use edgecase_core::graph::{Graph, Node, Edge};
use edgecase_core::algorithms::paths::shortest_path;

fn create_test_graph(n: usize) -> Graph {
    let mut graph = Graph::new(false, false);
    
    // Create nodes
    for i in 0..n {
        let node = Node::new(format!("Node{}", i), None);
        graph.add_node(node).unwrap();
    }
    
    // Create edges (chain)
    for i in 0..n-1 {
        let edge = Edge::new(
            format!("Node{}", i),
            format!("Node{}", i + 1),
            None,
            None,
        );
        graph.add_edge(edge).unwrap();
    }
    
    graph
}

fn bench_graph_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_creation");
    
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| create_test_graph(black_box(size)));
        });
    }
    
    group.finish();
}

fn bench_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_path");
    
    for size in [100, 500, 1000].iter() {
        let graph = create_test_graph(*size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                shortest_path(
                    black_box(&graph),
                    black_box("Node0"),
                    black_box(&format!("Node{}", size - 1)),
                )
            });
        });
    }
    
    group.finish();
}

fn bench_neighbors_lookup(c: &mut Criterion) {
    let graph = create_test_graph(10000);
    
    c.bench_function("neighbors_lookup", |b| {
        b.iter(|| graph.neighbors(black_box("Node5000")));
    });
}

criterion_group!(
    benches,
    bench_graph_creation,
    bench_shortest_path,
    bench_neighbors_lookup
);
criterion_main!(benches);
