//! Diagnostic: inspect Shape.step B-Rep topology — edge sharing, seam stitching, wire structure.
//! Run: cargo test -p rc3d-io --test shape_topology --release -- --nocapture

use rc3d_io::step::brep::build_brep;
use rc3d_io::step::brep::geom::{CurveGeom, SurfaceGeom};
use rc3d_shape::topo::{EdgeKey, FaceKey, VertexKey};
use rc3d_io::step::brep::heal::{auto_heal_shell, HealLevel};
use rc3d_io::step::parser;
use std::path::Path;
use std::collections::HashMap;

fn test_data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test_data")
        .join(name)
}

fn curve_name(c: &CurveGeom) -> &'static str {
    match c {
        CurveGeom::Line { .. } => "Line",
        CurveGeom::Circle { .. } => "Circ",
        CurveGeom::Trimmed { .. } => "Trim",
        CurveGeom::BSpline { .. } => "BSpl",
        _ => "Other",
    }
}

#[test]
fn inspect_shape_topology() {
    let path = test_data("Shape.step");
    let text = std::fs::read_to_string(&path).expect("read step");
    let exchange = parser::parse_exchange(&text).expect("parse");
    let brep = build_brep(&exchange.entities).expect("brep");
    let mut reg = brep.registry;
    for &sk in &brep.root_solids {
        if let Some(solid) = reg.solids.get(sk) {
            auto_heal_shell(solid.outer_shell, &mut reg, HealLevel::Standard, 5);
        }
    }

    println!("\n=== Topology Summary ===");
    println!("Vertices: {}  Edges: {}  Faces: {}  Wires: {}",
        reg.vertices.len(), reg.edges.len(), reg.faces.len(), reg.wires.len());

    // Per-edge: which faces reference it
    let mut edge_faces: HashMap<EdgeKey, Vec<FaceKey>> = HashMap::new();

    for &sk in &brep.root_solids {
        let Some(solid) = reg.solids.get(sk) else { continue; };
        let Some(shell) = reg.shells.get(solid.outer_shell) else { continue; };

        for &(face_key, _orient) in &shell.faces {
            let Some(face) = reg.faces.get(face_key) else { continue; };
            let sf_kind = match &face.surface {
                SurfaceGeom::Revolution { .. } => "Rev",
                SurfaceGeom::BSpline(_) => "BSpl",
                SurfaceGeom::Offset { .. } => "Offs",
                SurfaceGeom::Plane { .. } => "Plane",
                _ => "Other",
            };

            let mut wire_edges = Vec::new();
            for wk in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
                let Some(wire) = reg.wires.get(*wk) else { continue; };
                for (ek, orient) in &wire.edges {
                    wire_edges.push((*ek, *orient));
                    edge_faces.entry(*ek).or_default().push(face_key);
                }
            }

            println!("\nFace {:?} ({}) {} wire edges:",
                face_key, sf_kind, wire_edges.len());
            for (ek, orient) in &wire_edges {
                let edge = reg.edges.get(*ek).unwrap();
                let pc_count = edge.pcurves.len();
                let dir = if matches!(orient, rc3d_shape::topo::Orientation::Forward) { "F" } else { "R" };
                let ck = curve_name(&edge.curve);
                let shared = edge_faces.get(ek).map(|v| v.len()).unwrap_or(0);
                println!("  {:?} [{:?},{:?}] {} {} pc#{} by_{}",
                    ek, edge.v_low, edge.v_high, ck, dir, pc_count, shared);
            }
        }
    }

    // Shared vs unshared
    let shared: Vec<_> = edge_faces.iter().filter(|(_, fcs)| fcs.len() > 1).collect();
    let unshared: Vec<_> = edge_faces.iter().filter(|(_, fcs)| fcs.len() == 1).collect();
    println!("\n=== Edge Sharing: {} shared / {} unshared ===", shared.len(), unshared.len());

    if !shared.is_empty() {
        println!("-- Shared --");
        for (ek, faces) in &shared {
            let edge = reg.edges.get(**ek).unwrap();
            println!("  {:?}: faces={:?} curve={} pcurves#{}",
                ek, faces, curve_name(&edge.curve), edge.pcurves.len());
        }
    }

    if !unshared.is_empty() {
        println!("-- Unshared --");
        for (ek, faces) in &unshared {
            let edge = reg.edges.get(**ek).unwrap();
            println!("  {:?}: face {:?}, curve={}", ek, faces.first().unwrap(), curve_name(&edge.curve));
        }
    }

    // Debug: show full curve variant for "Other" edges
    println!("\n=== Other Curves Detail ===");
    for (ek, edge) in reg.edges.iter() {
        let disc = std::mem::discriminant(&edge.curve);
        let disc_val = match &edge.curve {
            rc3d_io::step::brep::geom::CurveGeom::Line { .. } => 0,
            rc3d_io::step::brep::geom::CurveGeom::Circle { .. } => 1,
            rc3d_io::step::brep::geom::CurveGeom::Ellipse { .. } => 2,
            rc3d_io::step::brep::geom::CurveGeom::Hyperbola { .. } => {
                println!("  {:?}: Hyperbola", ek);
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::Parabola { .. } => {
                println!("  {:?}: Parabola", ek);
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::BSpline { degree, control_points, knots, .. } => {
                println!("  {:?}: BSpline deg={} cp#{} knots#{}", ek, degree, control_points.len(), knots.len());
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::Trimmed { basis, t_min, t_max } => {
                println!("  {:?}: Trimmed({:?} t=[{:.3},{:.3}])", ek, std::mem::discriminant(basis.as_ref()), t_min, t_max);
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::Composite { .. } => {
                println!("  {:?}: Composite", ek);
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::Polyline { .. } => {
                println!("  {:?}: Polyline", ek);
                continue;
            }
            rc3d_io::step::brep::geom::CurveGeom::Offset { basis, offset_dir, distance } => {
                println!("  {:?}: Offset dist={} dir={:?}", ek, distance, offset_dir);
                continue;
            }
        };
        if disc_val > 1 { // not Line, not Circle
            println!("  {:?}: disc={:?}", ek, disc);
        }
    }

    // Seam edges
    println!("\n=== Seam Edges ===");
    for &sk in &brep.root_solids {
        let Some(solid) = reg.solids.get(sk) else { continue; };
        let Some(shell) = reg.shells.get(solid.outer_shell) else { continue; };
        for &(face_key, _) in &shell.faces {
            let Some(face) = reg.faces.get(face_key) else { continue; };
            if !face.seam_edges.is_empty() {
                println!("  Face {:?}: {:?}", face_key, face.seam_edges);
            }
        }
    }

    // Vertex sharing check: how many edges share each vertex?
    let mut vert_use: HashMap<VertexKey, usize> = HashMap::new();
    for ek in edge_faces.keys() {
        let Some(edge) = reg.edges.get(*ek) else { continue; };
        *vert_use.entry(edge.v_low).or_default() += 1;
        *vert_use.entry(edge.v_high).or_default() += 1;
    }
    let max_use = vert_use.values().max().copied().unwrap_or(0);
    println!("\n=== Vertex Valence: max edges/vertex = {} ===", max_use);
}
