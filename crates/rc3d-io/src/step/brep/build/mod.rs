//! STEP → B-Rep builder. T1.7-T1.9
//!
//! Converts STEP entity index into a parametric B-Rep registry.
//!
//! Architecture:
//!   Pass 1: Build SurfaceGeom for each face
//!   Pass 2: Build edges with PCURVEs per face
//!   Pass 3: Build wires, faces, shells, solids
//!   Pass 4: Assemble root_solids
//!
//! Chicken-and-egg: Need FaceKey to register PCURVEs, but need PCURVEs to build
//! the face. Solution: insert a placeholder face first, build edges with PCURVEs
//! using the real FaceKey, then update the face with correct wires.

use rc3d_core::math::Real;
mod shell;
mod surface;
mod pcurve;
mod curve;
pub mod tessellated;

pub use surface::build_surface;
pub use curve::build_curve;
pub use pcurve::{build_2d_curve, resolve_edge_pcurve};

use crate::step::parser::{EntityIndex, EntityRecord};
use crate::step::value::StepValue;
use crate::step::StepError;
use crate::step::entity_geom as geom;
use crate::step::topology;
use rc3d_shape::BRepStore;
use rc3d_shape::topo::*;
use super::geom::{CurveGeom, SurfaceGeom, plane_tangent_basis};
use rc3d_shape::geom::curve2d::Curve2d;
use super::geom::curve_eval::approx_chordal_length;
use super::geom::normalize_edge_curve_to_vertices;
use super::heal::curve_trim::add_degenerated_edge_at_pole;
use rc3d_core::math::PVec3;

#[derive(Debug, Default, Clone)]
pub struct BRepBuildReport {
    pub skipped_faces: usize,
    pub skipped_edges: usize,
    pub void_shell_count: usize,
    pub void_shells_subtracted: usize,
    pub oriented_forward_faces: usize,
    pub oriented_reversed_faces: usize,
    /// Unknown surfaces/curves substituted with plane/line (Preview fallback).
    pub geometry_fallback_count: usize,
}

#[derive(Debug, Clone)]
pub struct BRepBuildOptions {
    pub allow_geometry_fallback: bool,
    pub strict_voids: bool,
}

impl BRepBuildOptions {
    pub fn from_import(options: &crate::step::import_options::StepImportOptions) -> Self {
        Self {
            allow_geometry_fallback: options.allow_geometry_fallback(),
            strict_voids: options.strict_voids,
        }
    }
}

#[derive(Debug)]
pub struct BRepBuildResult {
    pub registry: BRepStore,
    pub root_solids: Vec<SolidKey>,
    pub build_report: BRepBuildReport,
}

/// Build a full B-Rep from STEP entities.
pub fn build_brep(entities: &EntityIndex) -> Result<BRepBuildResult, StepError> {
    build_brep_with_options(entities, &BRepBuildOptions {
        allow_geometry_fallback: true,
        strict_voids: false,
    })
}

struct ShellBuildCtx<'a> {
    reg: &'a mut BRepStore,
    entities: &'a EntityIndex,
    tol: Real,
    face_colors: &'a std::collections::HashMap<u64, [Real; 3]>,
    options: &'a BRepBuildOptions,
    skipped_faces: &'a mut usize,
    skipped_edges: &'a mut usize,
    oriented_forward_faces: &'a mut usize,
    oriented_reversed_faces: &'a mut usize,
    geometry_fallback_count: &'a mut usize,
}

fn resolve_face_surface(
    face_data: &topology::StepFace,
    entities: &EntityIndex,
    options: &BRepBuildOptions,
    skipped_faces: &mut usize,
    geometry_fallback_count: &mut usize,
) -> Option<(SurfaceGeom, Option<(Real, Real, Real, Real)>)> {
    if let Some(sid) = face_data.surface_id {
        if let Some(surface) = build_surface(sid, entities) {
            let trim = surface::build_surface_trim_range(sid, entities);
            return Some((surface, trim));
        }
        if options.allow_geometry_fallback {
            let surf_name = entities.get(&sid).map(|r| r.name.as_str()).unwrap_or("?");
            log::warn!(
                "[BRep] build_surface failed for #{} ({}), falling back to Plane from face edges",
                sid, surf_name
            );
            *geometry_fallback_count += 1;
            Some((fallback_plane_from_face(face_data), None))
        } else {
            *skipped_faces += 1;
            None
        }
    } else if options.allow_geometry_fallback {
        log::warn!("[BRep] face #{:?} has no surface reference, falling back to Plane from face edges",
            face_data.face_id);
        *geometry_fallback_count += 1;
        Some((fallback_plane_from_face(face_data), None))
    } else {
        *skipped_faces += 1;
        None
    }
}

/// Build a fallback plane from face edge vertices when the original surface cannot be resolved.
/// Uses Newell's method for an approximate normal and the vertex centroid as origin.
fn fallback_plane_from_face(face_data: &topology::StepFace) -> SurfaceGeom {
    let mut points: Vec<PVec3> = Vec::new();
    for bloop in &face_data.bounds {
        for edge in &bloop.edges {
            points.push(edge.start);
            points.push(edge.end);
        }
    }
    if points.is_empty() {
        return SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X };
    }

    // Centroid
    let inv_n = 1.0 / points.len() as Real;
    let origin = PVec3::new(
        points.iter().map(|p| p.x).sum::<Real>() * inv_n,
        points.iter().map(|p| p.y).sum::<Real>() * inv_n,
        points.iter().map(|p| p.z).sum::<Real>() * inv_n,
    );

    // Approximate normal via Newell's method (robust for non-planar polygons)
    let mut normal = PVec3::ZERO;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        normal.x += (points[i].y - points[j].y) * (points[i].z + points[j].z);
        normal.y += (points[i].z - points[j].z) * (points[i].x + points[j].x);
        normal.z += (points[i].x - points[j].x) * (points[i].y + points[j].y);
    }
    let normal_len = normal.length();
    let normal = if normal_len > 1e-10 { normal / normal_len } else { PVec3::Z };

    // u_dir from first edge direction
    let u_dir = if points.len() >= 2 {
        let d = points[1] - points[0];
        let dl = d.length();
        if dl > 1e-10 { d / dl } else { PVec3::X }
    } else {
        PVec3::X
    };
    let (u_dir, _v_dir) = plane_tangent_basis(normal, u_dir);

    SurfaceGeom::Plane { origin, normal, u_dir }
}

fn resolve_edge_curve(
    edge_data: &topology::StepEdge,
    entities: &EntityIndex,
    options: &BRepBuildOptions,
    skipped_edges: &mut usize,
    geometry_fallback_count: &mut usize,
) -> Option<CurveGeom> {
    if let Some(curve) = build_curve(edge_data.curve_id, entities) {
        // SURFACE_CURVE with a degenerate/short Line 3D curve may hide a Circle:
        // the actual circular geometry is defined by PCurves on the surface.
        if let Some(upgraded) = try_upgrade_to_circle(
            edge_data, &curve, entities,
        ) {
            return Some(upgraded);
        }
        return Some(curve);
    }
    if options.allow_geometry_fallback {
        *geometry_fallback_count += 1;
        let dir = edge_data.end - edge_data.start;
        let d = if dir.length() > 1e-10 { dir } else { PVec3::X };
        Some(CurveGeom::Line {
            origin: edge_data.start,
            direction: d,
        })
    } else {
        *skipped_edges += 1;
        None
    }
}

/// Detect when a SURFACE_CURVE's 3D curve is a degenerate/short Line hiding
/// a circle. The edge endpoints are far apart but the Line direction is near
/// zero → the edge is a circular arc on a cylinder/sphere.
///
/// Generates a Circle through the start/end points with axis perpendicular
/// to the edge chord, matching the cylinder/sphere radius.
fn try_upgrade_to_circle(
    edge_data: &topology::StepEdge,
    curve: &CurveGeom,
    entities: &EntityIndex,
) -> Option<CurveGeom> {
    // Only process SURFACE_CURVE with Line 3D curves
    let record = entities.get(&edge_data.curve_id)?;
    if record.name != "SURFACE_CURVE" {
        return None;
    }
    let line = match curve {
        CurveGeom::Line { origin, direction } => (origin, direction),
        _ => return None,
    };
    let line_len = line.1.length();
    let chord_len = (edge_data.end - edge_data.start).length();
    // Closed circle (start ≈ end) or very short 3D line relative to chord:
    // the real geometry is in the PCurves on the cylindrical/spherical surface.
    let is_closed = chord_len < line_len * 0.01 || chord_len < 1e-4;
    if !is_closed && line_len > chord_len * 0.1 {
        return None;
    }
    // Find the cylindrical/spherical surface among the PCURVEs
    let pcurve_ids = geom::nth_list_refs(&record.params, 2)?;
    for &pcurve_id in &pcurve_ids {
        let pc_record = entities.get(&pcurve_id)?;
        let surface_id = geom::nth_ref(&pc_record.params, 1)?;
        let surface = build_surface(surface_id, entities)?;
        match &surface {
            SurfaceGeom::Cylinder { origin, axis, radius, .. } => {
                let ax = axis.normalize();
                let r = *radius;
                // Project start/end onto plane perpendicular to axis through origin
                let mid = (edge_data.start + edge_data.end) * 0.5;
                let to_mid = mid - *origin;
                let axial = ax * ax.dot(to_mid);
                let radial = to_mid - axial;
                let center = *origin + axial + radial.normalize() * r;
                let (x_dir, y_dir) = rc3d_shape::geom::curve_eval::build_ortho_axes(ax);
                return Some(CurveGeom::Circle { center, axis: ax, radius: r, x_dir, y_dir });
            }
            SurfaceGeom::Sphere { center, radius: r } => {
                let mid = (edge_data.start + edge_data.end) * 0.5;
                let to_mid = mid - *center;
                let h = to_mid.length();
                if h > *r { continue; }
                let circ_r = (r * r - h * h).sqrt().max(1e-6);
                let axis = to_mid.normalize();
                let c = *center + axis * h;
                let (x_dir, y_dir) = rc3d_shape::geom::curve_eval::build_ortho_axes(axis);
                return Some(CurveGeom::Circle { center: c, axis, radius: circ_r, x_dir, y_dir });
            }
            _ => continue,
        }
    }
    None
}

/// Post-processing: iterate all edges of faces with cylindrical/spherical
/// surfaces. When an edge has a Line 3D curve but the face surface is curved,
/// replace the Line with a Circle computed from the surface + PCurve data.
fn upgrade_line_edges_to_circles(reg: &mut BRepStore) -> usize {
    let mut count = 0usize;
    let face_keys: Vec<FaceKey> = reg.faces.keys().collect();
    for fk in face_keys {
        let surface = match reg.faces.get(fk).map(|f| f.surface.clone()) {
            Some(s) => s, None => continue,
        };
        match &surface {
            SurfaceGeom::Cylinder { origin, axis, radius, .. } => {
                let ax = axis.normalize();
                let r = *radius;
                let edge_keys: Vec<EdgeKey> = rc3d_shape::topo_iter::iter_edges_of_face(fk, reg);
                for ek in edge_keys {
                    let edge = match reg.edges.get(ek) { Some(e) => e, None => continue };
                    if !matches!(edge.curve, CurveGeom::Line { .. }) { continue; }
                    if let Some(pc) = edge.pcurves.get(&fk) {
                        let uv_mid = pc.d0(0.5);
                        let v = uv_mid.1;
                        let center = *origin + ax * v;
                        let (x_dir, y_dir) = rc3d_shape::geom::curve_eval::build_ortho_axes(ax);
                        // Rebuild PCurve: fresh Line matching the Circle geometry
                        let fresh_pc = Curve2d::Line {
                            origin: (0.0, v),
                            direction: (std::f64::consts::TAU, 0.0),
                        };
                        if let Some(e) = reg.edges.get_mut(ek) {
                            e.curve = CurveGeom::Circle { center, axis: ax, radius: r, x_dir, y_dir };
                            e.pcurves.insert(fk, fresh_pc);
                            e.tolerance = 1e-4;
                            count += 1;
                        }
                    }
                }
            }
            SurfaceGeom::Sphere { center, radius: r } => {
                let edge_keys: Vec<EdgeKey> = rc3d_shape::topo_iter::iter_edges_of_face(fk, reg);
                for ek in edge_keys {
                    let edge = match reg.edges.get(ek) { Some(e) => e, None => continue };
                    if !matches!(edge.curve, CurveGeom::Line { .. }) { continue; }
                    if let Some(pc) = edge.pcurves.get(&fk) {
                        let uv = pc.d0(0.5);
                        let v_norm = uv.1 / std::f64::consts::PI;
                        let z = r * (1.0 - 2.0 * v_norm).cos();
                        let circ_r = (r * r - z * z).sqrt().max(1e-6);
                        let axis = PVec3::Z;
                        let c = *center + axis * z;
                        let (x_dir, y_dir) = rc3d_shape::geom::curve_eval::build_ortho_axes(axis);
                        // Rebuild PCurve: fresh Line for the latitude circle
                        let fresh_pc = Curve2d::Line {
                            origin: (0.0, uv.1),
                            direction: (std::f64::consts::TAU, 0.0),
                        };
                        if let Some(e) = reg.edges.get_mut(ek) {
                            e.curve = CurveGeom::Circle { center: c, axis, radius: circ_r, x_dir, y_dir };
                            e.pcurves.insert(fk, fresh_pc);
                            e.tolerance = 1e-4;
                            count += 1;
                        }
                    }
                }
            }
            _ => continue,
        }
    }
    count
}

pub fn build_brep_with_options(
    entities: &EntityIndex,
    options: &BRepBuildOptions,
) -> Result<BRepBuildResult, StepError> {
    let solid_models = topology::collect_solid_models(entities);
    if solid_models.is_empty() {
        // Fallback: try tessellated geometry (AP242 TESSELLATED_SHELL)
        let tess_shells = tessellated::collect_tessellated_shells(entities);
        if tess_shells.is_empty() {
            return Err(StepError::NoGeometry);
        }
        log::info!(
            "[STEP] No B-Rep shells found, using {} tessellated shell(s) as fallback",
            tess_shells.len()
        );
        return build_tessellated_fallback(entities, &tess_shells);
    }

    let tol = topology::global_tolerance(entities);
    let mut reg = BRepStore::with_tolerance(rc3d_shape::ToleranceContext::from_model(tol));
    let face_colors = topology::collect_face_colors(entities);
    let mut skipped_faces = 0usize;
    let mut skipped_edges = 0usize;
    let mut oriented_forward_faces = 0usize;
    let mut oriented_reversed_faces = 0usize;
    let mut geometry_fallback_count = 0usize;
    let mut root_solids = Vec::new();

    for model in &solid_models {
        let mut ctx = ShellBuildCtx {
            reg: &mut reg,
            entities,
            tol,
            face_colors: &face_colors,
            options,
            skipped_faces: &mut skipped_faces,
            skipped_edges: &mut skipped_edges,
            oriented_forward_faces: &mut oriented_forward_faces,
            oriented_reversed_faces: &mut oriented_reversed_faces,
            geometry_fallback_count: &mut geometry_fallback_count,
        };

        let outer_shell = match shell::build_shell_from_step(&model.outer, &mut ctx) {
            Some(sk) => sk,
            None => continue,
        };

        let void_shells: Vec<ShellKey> = model
            .voids
            .iter()
            .filter_map(|void_shell| shell::build_shell_from_step(void_shell, &mut ctx))
            .collect();

        let solid_key = reg.solids.insert(BRepSolid {
            outer_shell,
            void_shells,
        });
        root_solids.push(solid_key);
    }

    if root_solids.is_empty() {
        return Err(StepError::NoGeometry);
    }

    // Build edge-to-face inverted index for fast shared edge queries
    reg.build_edge_to_faces_index();

    // Post-process: upgrade degenerate Line edges on cylindrical/spherical
    // faces to Circle curves.
    let upgraded = upgrade_line_edges_to_circles(&mut reg);
    if upgraded > 0 {
        log::info!("[STEP] Upgraded {} Line edges to Circle on curved surfaces", upgraded);
    }

    // Post-process: add seam edges to closed parametric surfaces (Sphere, Cylinder).
    // These are needed for topological validity; without them OCC-based tools
    // may reject the geometry.
    let mut seams_added = 0usize;
    for fk in reg.faces.keys().collect::<Vec<_>>() {
        seams_added += rc3d_shape::heal::seam::fix_missing_seams(&mut reg, fk);
    }
    if seams_added > 0 {
        log::info!("[STEP] Added {} seam edges to closed surfaces", seams_added);
    }

    // Phase: Void shell subtraction (when strict_voids enabled)
    let mut void_shells_subtracted = 0usize;
    if options.strict_voids {
        for &solid_key in &root_solids {
            void_shells_subtracted += subtract_void_shells(solid_key, &mut reg);
        }
    }

    let void_shell_count = root_solids
        .iter()
        .filter_map(|&sk| reg.solids.get(sk))
        .map(|s| s.void_shells.len())
        .sum();

    Ok(BRepBuildResult {
        registry: reg,
        root_solids,
        build_report: BRepBuildReport {
            skipped_faces,
            skipped_edges,
            void_shell_count,
            void_shells_subtracted,
            oriented_forward_faces,
            oriented_reversed_faces,
            geometry_fallback_count,
        },
    })
}

/// Hole-punching: project void shell faces onto outer shell as inner wires.
///
/// For each void face, finds the matching outer face via shared edge lookup
/// in `edge_to_faces`, then copies void wire edges as an inner wire (hole).
/// Works for cylindrical holes, counterbores, and pockets where void and outer
/// faces share a boundary edge.
fn subtract_void_shells(solid_key: SolidKey, reg: &mut BRepStore) -> usize {
    use std::collections::HashMap;

    let void_shells = {
        let solid = match reg.solids.get(solid_key) {
            Some(s) => s,
            None => return 0,
        };
        let outer_faces: Vec<FaceKey> = reg
            .shells
            .get(solid.outer_shell)
            .map(|s| s.faces.iter().map(|(f, _)| *f).collect())
            .unwrap_or_default();
        (solid.void_shells.clone(), outer_faces)
    };
    let (void_shell_keys, outer_face_keys) = void_shells;

    let mut holes = 0usize;
    for &void_sk in &void_shell_keys {
        let void_faces: Vec<(FaceKey, Vec<EdgeKey>)> = {
            let vs = match reg.shells.get(void_sk) {
                Some(s) => s,
                None => continue,
            };
            vs.faces
                .iter()
                .map(|(fk, _)| {
                    let edges: Vec<EdgeKey> = reg
                        .wires
                        .get(
                            reg.faces
                                .get(*fk)
                                .map(|f| f.outer_wire)
                                .unwrap_or_default(),
                        )
                        .map(|w| w.edges.iter().map(|(ek, _)| *ek).collect())
                        .unwrap_or_default();
                    (*fk, edges)
                })
                .collect()
        };

        for (void_fk, void_edge_keys) in &void_faces {
            // Vote: which outer face shares the most edges with this void face?
            let mut scores: HashMap<FaceKey, usize> = HashMap::new();
            for ek in void_edge_keys {
                if let Some(face_list) = reg.edge_to_faces.get(ek) {
                    for fk in face_list {
                        if *fk != *void_fk && outer_face_keys.contains(fk) {
                            *scores.entry(*fk).or_default() += 1;
                        }
                    }
                }
            }
            let best = scores.into_iter().max_by_key(|(_, c)| *c).map(|(fk, _)| fk);
            if let Some(outer_fk) = best {
                if punch_hole(outer_fk, *void_fk, reg) {
                    holes += 1;
                }
            }
        }
    }
    holes
}

/// Copy void face wire as an inner wire on the matching outer face.
fn punch_hole(outer_fk: FaceKey, void_fk: FaceKey, reg: &mut BRepStore) -> bool {
    let void_wire_key = match reg.faces.get(void_fk) {
        Some(f) => f.outer_wire,
        None => return false,
    };
    let void_wire_edges = match reg.wires.get(void_wire_key) {
        Some(w) => w.edges.clone(),
        None => return false,
    };
    if void_wire_edges.is_empty() {
        return false;
    }

    let inner_wire_key = reg.wires.insert(BRepWire {
        edges: void_wire_edges,
    });
    if let Some(face) = reg.faces.get_mut(outer_fk) {
        face.inner_wires.push(inner_wire_key);
        return true;
    }
    false
}

/// Fallback: build a minimal B-Rep from tessellated shells (AP242 pure-tessellation files).
/// Each tessellated shell becomes a single solid with one face per mesh.
fn build_tessellated_fallback(
    entities: &EntityIndex,
    tess_shells: &std::collections::HashMap<u64, Vec<tessellated::TessellatedMesh>>,
) -> Result<BRepBuildResult, StepError> {
    let tol = topology::global_tolerance(entities);
    let mut reg = BRepStore::with_tolerance(rc3d_shape::ToleranceContext::from_model(tol));
    let mut root_solids = Vec::new();

    for (&_shell_id, meshes) in tess_shells {
        let mut face_keys = Vec::new();
        for mesh in meshes {
            // Create a placeholder face with plane surface and empty wire.
            // The actual triangle data is stored in the mesh pipeline separately.
            let wire_key = reg.wires.insert(BRepWire { edges: vec![] });
            let face_key = reg.faces.insert(BRepFace {
                surface: SurfaceGeom::Plane {
                    origin: PVec3::ZERO,
                    normal: PVec3::Z,
                    u_dir: PVec3::X,
                },
                outer_wire: wire_key,
                inner_wires: vec![],
                same_sense: true,
                tolerance: tol,
                seam_edges: vec![],
                color: mesh.color,
                degenerated_edges: vec![],
            });
            face_keys.push((face_key, Orientation::Forward));
        }
        if !face_keys.is_empty() {
            let shell_key = reg.shells.insert(BRepShell {
                faces: face_keys,
                closed: false,
                step_id: None,
            });
            let solid_key = reg.solids.insert(BRepSolid {
                outer_shell: shell_key,
                void_shells: vec![],
            });
            root_solids.push(solid_key);
        }
    }

    if root_solids.is_empty() {
        return Err(StepError::NoGeometry);
    }

    Ok(BRepBuildResult {
        registry: reg,
        root_solids,
        build_report: BRepBuildReport::default(),
    })
}
