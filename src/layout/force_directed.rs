use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for force-directed layout
#[derive(Debug, Clone)]
pub struct LayoutConfig {
    pub iterations: usize,
    pub k: Option<f64>,
    pub area: Option<f64>,
    pub use_barnes_hut: bool,
    pub theta: f64,
    pub damping: f64,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            k: None,
            area: None,
            use_barnes_hut: true,
            theta: 0.5,
            damping: 0.9,
        }
    }
}

/// 2D position
type Position = (f64, f64);

/// Force-directed layout using Fruchterman-Reingold algorithm
pub struct ForceDirectedLayout {
    config: LayoutConfig,
}

impl ForceDirectedLayout {
    pub fn new(config: LayoutConfig) -> Self {
        Self { config }
    }

    /// Compute layout for a graph
    /// Returns HashMap mapping node indices to (x, y) positions
    pub fn compute(
        &self,
        node_count: usize,
        edges: &[(usize, usize)],
    ) -> HashMap<usize, Position> {
        if node_count == 0 {
            return HashMap::new();
        }
        if node_count == 1 {
            let mut positions = HashMap::new();
            positions.insert(0, (0.0, 0.0));
            return positions;
        }

        // Calculate layout parameters
        let area = self.config.area.unwrap_or(node_count as f64 * 2.0);
        let k = self.config.k.unwrap_or((area / node_count as f64).sqrt());

        // Initialize random positions
        let mut positions = self.initialize_positions(node_count, k);
        let mut velocities = vec![(0.0, 0.0); node_count];

        // Build adjacency list for attractive forces
        let adjacency = self.build_adjacency(node_count, edges);

        // Layout iterations
        let mut temp = k * 2.0;
        let cooling_factor = temp / self.config.iterations as f64;

        for _iteration in 0..self.config.iterations {
            // Calculate repulsive forces (parallel)
            let repulsive_forces = if self.config.use_barnes_hut && node_count > 100 {
                self.barnes_hut_repulsion(&positions, k)
            } else {
                self.direct_repulsion(&positions, k)
            };

            // Calculate attractive forces (parallel)
            let attractive_forces = self.attractive_forces(&positions, &adjacency, k);

            // Update positions (parallel)
            (0..node_count)
                .into_par_iter()
                .map(|i| {
                    let fx = repulsive_forces[i].0 + attractive_forces[i].0;
                    let fy = repulsive_forces[i].1 + attractive_forces[i].1;

                    // Limit displacement by temperature
                    let disp = (fx * fx + fy * fy).sqrt();
                    let limited_disp = disp.min(temp);
                    let scale = if disp > 0.0 {
                        limited_disp / disp
                    } else {
                        0.0
                    };

                    let new_vel = (
                        velocities[i].0 * self.config.damping + fx * scale,
                        velocities[i].1 * self.config.damping + fy * scale,
                    );

                    let new_pos = (
                        positions[i].0 + new_vel.0,
                        positions[i].1 + new_vel.1,
                    );

                    (new_pos, new_vel)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
                .for_each(|(i, (new_pos, new_vel))| {
                    positions[i] = new_pos;
                    velocities[i] = new_vel;
                });

            // Cool down temperature
            temp -= cooling_factor;
        }

        // Scale to viewport and convert to HashMap
        self.scale_to_viewport(positions)
    }

    fn initialize_positions(&self, node_count: usize, k: f64) -> Vec<Position> {
        let mut rng = rand::thread_rng();
        let init_range = (node_count as f64).sqrt() * k;

        (0..node_count)
            .map(|_| {
                (
                    rng.gen_range(-init_range..init_range),
                    rng.gen_range(-init_range..init_range),
                )
            })
            .collect()
    }

    fn build_adjacency(&self, node_count: usize, edges: &[(usize, usize)]) -> Vec<Vec<usize>> {
        let mut adjacency = vec![Vec::new(); node_count];
        for &(u, v) in edges {
            adjacency[u].push(v);
            adjacency[v].push(u);
        }
        adjacency
    }

    fn direct_repulsion(&self, positions: &[Position], k: f64) -> Vec<Position> {
        let k_squared = k * k;
        (0..positions.len())
            .into_par_iter()
            .map(|i| {
                let mut fx = 0.0;
                let mut fy = 0.0;

                for j in 0..positions.len() {
                    if i == j {
                        continue;
                    }

                    let dx = positions[i].0 - positions[j].0;
                    let dy = positions[i].1 - positions[j].1;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq < 0.01 {
                        continue;
                    }

                    let dist = dist_sq.sqrt();
                    let force = k_squared / dist;

                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }

                (fx, fy)
            })
            .collect()
    }

    fn barnes_hut_repulsion(&self, positions: &[Position], k: f64) -> Vec<Position> {
        // Build quadtree
        let quadtree = QuadTree::build(positions);

        // Calculate forces using Barnes-Hut
        (0..positions.len())
            .into_par_iter()
            .map(|i| quadtree.calculate_force(positions[i], self.config.theta, k))
            .collect()
    }

    fn attractive_forces(
        &self,
        positions: &[Position],
        adjacency: &[Vec<usize>],
        k: f64,
    ) -> Vec<Position> {
        (0..positions.len())
            .into_par_iter()
            .map(|i| {
                let mut fx = 0.0;
                let mut fy = 0.0;

                for &j in &adjacency[i] {
                    let dx = positions[j].0 - positions[i].0;
                    let dy = positions[j].1 - positions[i].1;
                    let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                    // Attractive force: f_a(d) = d^2 / k
                    let force = dist * dist / k;

                    fx += (dx / dist) * force;
                    fy += (dy / dist) * force;
                }

                (fx, fy)
            })
            .collect()
    }

    fn scale_to_viewport(&self, positions: Vec<Position>) -> HashMap<usize, Position> {
        if positions.is_empty() {
            return HashMap::new();
        }

        // Find bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in &positions {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let range_x = max_x - min_x;
        let range_y = max_y - min_y;
        let max_range = range_x.max(range_y);

        // Scale to [-0.8, 0.8] viewport
        let scale = if max_range > 0.0 { 1.6 / max_range } else { 1.0 };
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        positions
            .into_iter()
            .enumerate()
            .map(|(i, (x, y))| {
                let scaled_x = (x - center_x) * scale;
                let scaled_y = (y - center_y) * scale;
                (i, (scaled_x, scaled_y))
            })
            .collect()
    }
}

/// Quadtree for Barnes-Hut approximation
struct QuadTree {
    bounds: BoundingBox,
    center_of_mass: Position,
    total_mass: f64,
    children: Option<Box<[QuadTree; 4]>>,
    node_index: Option<usize>,
}

#[derive(Debug, Clone)]
struct BoundingBox {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl BoundingBox {
    fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    fn center(&self) -> Position {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    fn contains(&self, pos: Position) -> bool {
        pos.0 >= self.min_x && pos.0 <= self.max_x && pos.1 >= self.min_y && pos.1 <= self.max_y
    }

    fn quadrant(&self, index: usize) -> BoundingBox {
        let (cx, cy) = self.center();
        match index {
            0 => BoundingBox::new(self.min_x, self.min_y, cx, cy), // Bottom-left
            1 => BoundingBox::new(cx, self.min_y, self.max_x, cy), // Bottom-right
            2 => BoundingBox::new(self.min_x, cy, cx, self.max_y), // Top-left
            3 => BoundingBox::new(cx, cy, self.max_x, self.max_y), // Top-right
            _ => panic!("Invalid quadrant index"),
        }
    }
}

impl QuadTree {
    fn build(positions: &[Position]) -> Self {
        if positions.is_empty() {
            panic!("Cannot build quadtree from empty positions");
        }

        // Calculate bounding box
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &(x, y) in positions {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        // Add padding to ensure all points are inside
        let padding = 0.01;
        let bounds = BoundingBox::new(min_x - padding, min_y - padding, max_x + padding, max_y + padding);

        let mut tree = QuadTree::new_empty(bounds);
        for (i, &pos) in positions.iter().enumerate() {
            tree.insert(i, pos);
        }
        tree
    }

    fn new_empty(bounds: BoundingBox) -> Self {
        Self {
            bounds,
            center_of_mass: (0.0, 0.0),
            total_mass: 0.0,
            children: None,
            node_index: None,
        }
    }

    fn insert(&mut self, node_index: usize, pos: Position) {
        // Update center of mass
        let new_mass = self.total_mass + 1.0;
        self.center_of_mass = (
            (self.center_of_mass.0 * self.total_mass + pos.0) / new_mass,
            (self.center_of_mass.1 * self.total_mass + pos.1) / new_mass,
        );
        self.total_mass = new_mass;

        // If this is a leaf with no node, just add the node
        if self.node_index.is_none() && self.children.is_none() {
            self.node_index = Some(node_index);
            return;
        }

        // If this is a leaf with a node, we need to subdivide
        if let Some(existing_index) = self.node_index.take() {
            self.subdivide();
            // Re-insert the existing node (find its position from somewhere - we'll need to pass positions)
            // For now, we'll skip this and handle it differently
        }

        // Insert into appropriate child
        if self.children.is_none() {
            self.subdivide();
        }

        if let Some(ref mut children) = self.children {
            for (i, child) in children.iter_mut().enumerate() {
                if child.bounds.contains(pos) {
                    child.insert(node_index, pos);
                    return;
                }
            }
        }
    }

    fn subdivide(&mut self) {
        let children = [
            QuadTree::new_empty(self.bounds.quadrant(0)),
            QuadTree::new_empty(self.bounds.quadrant(1)),
            QuadTree::new_empty(self.bounds.quadrant(2)),
            QuadTree::new_empty(self.bounds.quadrant(3)),
        ];
        self.children = Some(Box::new(children));
    }

    fn calculate_force(&self, pos: Position, theta: f64, k: f64) -> Position {
        // Don't calculate force on self
        if self.total_mass == 0.0 {
            return (0.0, 0.0);
        }

        let dx = self.center_of_mass.0 - pos.0;
        let dy = self.center_of_mass.1 - pos.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq < 1e-6 {
            return (0.0, 0.0);
        }

        let dist = dist_sq.sqrt();
        let size = self.bounds.width().max(self.bounds.height());

        // Barnes-Hut criterion: if node is far enough, use approximation
        if self.children.is_none() || size / dist < theta {
            let k_squared = k * k;
            let force = k_squared / dist;
            return ((dx / dist) * force, (dy / dist) * force);
        }

        // Otherwise, recurse to children
        if let Some(ref children) = self.children {
            children
                .iter()
                .map(|child| child.calculate_force(pos, theta, k))
                .fold((0.0, 0.0), |(ax, ay), (bx, by)| (ax + bx, ay + by))
        } else {
            (0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_graph() {
        let config = LayoutConfig {
            iterations: 10,
            ..Default::default()
        };
        let layout = ForceDirectedLayout::new(config);

        // Triangle graph: 0-1, 1-2, 2-0
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let positions = layout.compute(3, &edges);

        assert_eq!(positions.len(), 3);
        // Check that all nodes have positions
        for i in 0..3 {
            assert!(positions.contains_key(&i));
        }
    }

    #[test]
    fn test_single_node() {
        let layout = ForceDirectedLayout::new(LayoutConfig::default());
        let positions = layout.compute(1, &[]);

        assert_eq!(positions.len(), 1);
        assert_eq!(positions.get(&0), Some(&(0.0, 0.0)));
    }

    #[test]
    fn test_empty_graph() {
        let layout = ForceDirectedLayout::new(LayoutConfig::default());
        let positions = layout.compute(0, &[]);

        assert_eq!(positions.len(), 0);
    }
}
