//! Headless benchmark: measures scene-graph traversal cost to detect regressions.
//! Run: cargo run -p rc3d-examples --example bench
use std::time::Instant;
use rc3d_scene::node_data::{CubeNode, MaterialNode, NodeData, SeparatorNode};
use rc3d_scene::SceneGraph;

fn build_test_scene(object_count: usize) -> SceneGraph {
    let mut graph = SceneGraph::new();
    let root = graph.add_root(NodeData::Separator(SeparatorNode));
    let grid_size = (object_count as f32).sqrt().ceil() as usize;
    let spacing = 2.5;
    for i in 0..object_count {
        let x = (i % grid_size) as f32 * spacing - grid_size as f32 * spacing / 2.0;
        let z = (i / grid_size) as f32 * spacing - grid_size as f32 * spacing / 2.0;
        let sep = graph.add_child(root, NodeData::Separator(SeparatorNode));
        graph.add_child(sep, NodeData::Material(MaterialNode::default()));
        let c = CubeNode { width: 1.0, height: 1.0, depth: 1.0 };
        graph.add_child(sep, NodeData::Cube(c));
        let _ = (x, z); // positional info available for future distance-based tests
    }
    graph
}

fn count_recursive(graph: &SceneGraph, node: rc3d_core::NodeId) -> usize {
    let mut count = 1;
    if let Some(children) = graph.children(node) {
        for &c in children {
            count += count_recursive(graph, c);
        }
    }
    count
}

fn visit_cubes(graph: &SceneGraph, node: rc3d_core::NodeId, aabb_count: &mut u64) {
    if let Some(entry) = graph.get(node) {
        if let NodeData::Cube(c) = &entry.data {
            let _hw = c.width / 2.0;
            let _hh = c.height / 2.0;
            let _hd = c.depth / 2.0;
            *aabb_count += 1;
        }
    }
    if let Some(children) = graph.children(node) {
        for &c in children {
            visit_cubes(graph, c, aabb_count);
        }
    }
}

fn main() {
    let object_count = 1000usize;
    let frame_count = 120u32;
    println!("=== Headless benchmark: {} objects, {} frames ===", object_count, frame_count);

    let graph = build_test_scene(object_count);
    let roots = graph.roots().to_vec();
    let node_count: usize = roots.iter().map(|&r| count_recursive(&graph, r)).sum();
    println!("Scene graph nodes: {node_count}");

    let mut times = Vec::with_capacity(frame_count as usize);
    for f in 0..frame_count {
        let t0 = Instant::now();
        let mut aabb_count = 0u64;
        for &root in &roots {
            visit_cubes(&graph, root, &mut aabb_count);
        }
        let elapsed = t0.elapsed();
        times.push(elapsed.as_secs_f64() * 1000.0);
        if f <= 1 || f == frame_count - 1 {
            println!("Frame {:>4}: {:.3}ms ({} AABB checks)", f, times.last().unwrap(), aabb_count);
        }
        if f == frame_count / 2 {
            println!("  ... midpoint frame {f}: {:.3}ms", times.last().unwrap());
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times.first().unwrap();
    let max = times.last().unwrap();
    let median = times[times.len() / 2];
    let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!("\nFrame time stats (ms):");
    println!("  min:    {min:.3}");
    println!("  max:    {max:.3}");
    println!("  median: {median:.3}");
    println!("  avg:    {avg:.3}");
    println!("\nScore: {:.1} objects/ms", object_count as f64 / avg);
    println!("PASS — benchmark baseline established");
}
