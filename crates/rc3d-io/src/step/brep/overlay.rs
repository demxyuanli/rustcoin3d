//! B-Rep edge curve overlay for the scene graph.

use std::collections::HashSet;

use rc3d_core::math::Vec3;
use rc3d_scene::SceneGraph;
use rc3d_scene::node_data::{
    AnnotationNode, Coordinate3Node, IndexedLineSetNode, SeparatorNode, TransformNode,
};

use crate::step::assembly::AssemblyTransform;
use crate::step::brep::geom::{CurveGeom, SurfaceGeom};
use crate::step::brep::mesh::edge_disc::discretize_all_edges;
use crate::step::brep::mesh::BRepMeshConfig;
use crate::step::brep::registry::BRepRegistry;
use crate::step::brep::topo::{EdgeKey, FaceKey, ShellKey, SolidKey};

/// Pull seam/overlay lines slightly along the outward normal to avoid depth fighting.
const SURFACE_NORMAL_BIAS_FRAC: f32 = 3e-5;

/// Build an `IndexedLineSet` overlay from B-Rep edge curves.
///
/// Returns the transform node used for visibility toggling (scale to zero hides lines).
pub fn build_edge_curves(
    graph: &mut SceneGraph,
    root: rc3d_core::NodeId,
    reg: &BRepRegistry,
    root_solids: &[SolidKey],
    shell_transforms: &std::collections::HashMap<u64, AssemblyTransform>,
    mesh_config: &BRepMeshConfig,
) -> Option<rc3d_core::NodeId> {
    let edge_polys = discretize_all_edges(reg, &mesh_config.edge);

    let mut points: Vec<Vec3> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();

    for &sk in root_solids {
        let solid = reg.solids.get(sk)?;
        let shell = reg.shells.get(solid.outer_shell)?;
        let xform = shell
            .step_id
            .and_then(|id| shell_transforms.get(&id));

        let (edge_keys, seam_keys) = collect_shell_edge_keys(reg, solid.outer_shell);
        append_edge_segments(
            reg,
            &edge_polys,
            &edge_keys,
            &seam_keys,
            xform,
            &mut points,
            &mut indices,
        );
    }

    if points.is_empty() || indices.is_empty() {
        return None;
    }

    log::info!(
        "[STEP] edge overlay: {} segments, {} points",
        indices.len() / 2,
        points.len()
    );

    let sep = graph.add_child(root, rc3d_scene::NodeData::Separator(SeparatorNode));
    if let Some(entry) = graph.get_mut(sep) {
        entry.name = Some("BRepEdges".to_string());
    }

    let xform = graph.add_child(
        sep,
        rc3d_scene::NodeData::Transform(TransformNode::default()),
    );
    let annot = graph.add_child(
        xform,
        rc3d_scene::NodeData::Annotation(AnnotationNode),
    );
    graph.add_child(
        annot,
        rc3d_scene::NodeData::Coordinate3(Coordinate3Node { point: points }),
    );
    graph.add_child(
        annot,
        rc3d_scene::NodeData::IndexedLineSet(IndexedLineSetNode {
            coord_index: indices,
            line_width: 1.5,
            color: [0.15, 0.45, 1.0, 1.0],
        }),
    );

    Some(xform)
}

fn collect_shell_edge_keys(
    reg: &BRepRegistry,
    shell_key: ShellKey,
) -> (HashSet<EdgeKey>, HashSet<EdgeKey>) {
    let mut keys = HashSet::new();
    let mut seams = HashSet::new();
    let Some(shell) = reg.shells.get(shell_key) else {
        return (keys, seams);
    };

    for &(face_key, _) in &shell.faces {
        let Some(face) = reg.faces.get(face_key) else {
            continue;
        };
        if let Some(wire) = reg.wires.get(face.outer_wire) {
            for &(ek, _) in &wire.edges {
                keys.insert(ek);
            }
        }
        for &iw in &face.inner_wires {
            if let Some(wire) = reg.wires.get(iw) {
                for &(ek, _) in &wire.edges {
                    keys.insert(ek);
                }
            }
        }
        for &ek in &face.seam_edges {
            keys.insert(ek);
            seams.insert(ek);
        }
    }
    (keys, seams)
}

fn append_edge_segments(
    reg: &BRepRegistry,
    edge_polys: &std::collections::HashMap<EdgeKey, crate::step::brep::mesh::edge_disc::EdgePolygon>,
    edge_keys: &HashSet<EdgeKey>,
    seam_keys: &HashSet<EdgeKey>,
    xform: Option<&AssemblyTransform>,
    points: &mut Vec<Vec3>,
    indices: &mut Vec<i32>,
) {
    for &ek in edge_keys {
        let Some(poly) = edge_polys.get(&ek) else {
            continue;
        };
        if poly.params_3d.len() < 2 {
            continue;
        }

        let mut chain: Vec<Vec3> = poly
            .params_3d
            .iter()
            .map(|(_, p)| {
                xform
                    .map(|t| t.transform_point(*p))
                    .unwrap_or(*p)
            })
            .collect();

        if seam_keys.contains(&ek) {
            pull_chain_on_surface(reg, ek, &mut chain);
        }

        dedup_consecutive(&mut chain, 1e-7);

        let closed = is_closed_edge(reg, ek);
        if closed {
            append_closed_polyline_chain(&chain, points, indices);
        } else {
            append_polyline_chain(&chain, points, indices);
        }
    }
}

fn is_closed_edge(reg: &BRepRegistry, ek: EdgeKey) -> bool {
    reg.edges
        .get(ek)
        .map(|e| e.v_low == e.v_high)
        .unwrap_or(false)
}

/// Offset seam polyline along the face outward normal so it renders in front of the mesh.
fn pull_chain_on_surface(reg: &BRepRegistry, ek: EdgeKey, chain: &mut [Vec3]) {
    let Some(edge) = reg.edges.get(ek) else {
        return;
    };
    let Some((face_key, pcurve)) = primary_face_pcurve(edge) else {
        return;
    };
    let Some(face) = reg.faces.get(face_key) else {
        return;
    };

    let n_pts = chain.len();
    let denom = (n_pts - 1).max(1) as f32;

    for (i, p) in chain.iter_mut().enumerate() {
        let t = i as f32 / denom;
        let uv = pcurve.d0(t);
        let mut n = face.surface.normal_native(uv.x, uv.y);
        if !face.same_sense {
            n = -n;
        }
        let bias = surface_normal_bias(&face.surface, *p);
        *p += n * bias;
    }
}

fn primary_face_pcurve(edge: &crate::step::brep::topo::BRepEdge) -> Option<(FaceKey, &CurveGeom)> {
    let mut keys: Vec<FaceKey> = edge.pcurves.keys().copied().collect();
    keys.sort_unstable();
    let face_key = *keys.first()?;
    let pcurve = edge.pcurves.get(&face_key)?;
    Some((face_key, pcurve))
}

fn surface_normal_bias(surface: &SurfaceGeom, pos: Vec3) -> f32 {
    let scale = match surface {
        SurfaceGeom::Sphere { radius, .. } => *radius,
        SurfaceGeom::Cylinder { radius, .. } => *radius,
        SurfaceGeom::Torus { major_r, minor_r, .. } => major_r + minor_r,
        _ => pos.length().max(1.0),
    };
    (scale * SURFACE_NORMAL_BIAS_FRAC).max(1e-6)
}

fn dedup_consecutive(chain: &mut Vec<Vec3>, tol: f32) {
    chain.dedup_by(|a, b| (*a - *b).length() < tol);
}

/// Append an open polyline using shared vertex indices.
fn append_polyline_chain(chain: &[Vec3], points: &mut Vec<Vec3>, indices: &mut Vec<i32>) {
    if chain.len() < 2 {
        return;
    }
    let base = points.len() as i32;
    for p in chain {
        points.push(*p);
    }
    for i in 0..chain.len() - 1 {
        indices.push(base + i as i32);
        indices.push(base + i as i32 + 1);
    }
}

/// Append a closed polyline (last vertex connects back to first).
fn append_closed_polyline_chain(chain: &[Vec3], points: &mut Vec<Vec3>, indices: &mut Vec<i32>) {
    if chain.len() < 2 {
        return;
    }
    append_polyline_chain(chain, points, indices);
    let base = points.len() as i32 - chain.len() as i32;
    let last = base + chain.len() as i32 - 1;
    indices.push(last);
    indices.push(base);
}
