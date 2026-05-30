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

mod shell;
mod surface;
mod pcurve;
mod curve;

pub use surface::build_surface;
pub use curve::build_curve;
pub use pcurve::{build_2d_curve, resolve_edge_pcurve};

use crate::step::parser::{EntityIndex, EntityRecord};
use crate::step::value::StepValue;
use crate::step::StepError;
use crate::step::geom;
use crate::step::topology;
use crate::step::nurbs::NurbsSurface;
use super::registry::BRepRegistry;
use super::topo::*;
use super::geom::{CurveGeom, SurfaceGeom, plane_tangent_basis};
use super::geom::normalize_edge_curve_to_vertices;
use super::heal::curve_trim::add_degenerated_edge_at_pole;
use rc3d_core::math::Vec3;

#[derive(Debug, Default, Clone)]
pub struct BRepBuildReport {
    pub skipped_faces: usize,
    pub skipped_edges: usize,
    pub void_shell_count: usize,
}

#[derive(Debug, Clone)]
pub struct BRepBuildOptions {
    pub allow_geometry_fallback: bool,
}

impl BRepBuildOptions {
    pub fn from_import(options: &crate::step::import_options::StepImportOptions) -> Self {
        Self {
            allow_geometry_fallback: options.allow_geometry_fallback(),
        }
    }
}

#[derive(Debug)]
pub struct BRepBuildResult {
    pub registry: BRepRegistry,
    pub root_solids: Vec<SolidKey>,
    pub build_report: BRepBuildReport,
}

/// Build a full B-Rep from STEP entities.
pub fn build_brep(entities: &EntityIndex) -> Result<BRepBuildResult, StepError> {
    build_brep_with_options(entities, &BRepBuildOptions {
        allow_geometry_fallback: true,
    })
}

struct ShellBuildCtx<'a> {
    reg: &'a mut BRepRegistry,
    entities: &'a EntityIndex,
    tol: f32,
    face_colors: &'a std::collections::HashMap<u64, [f32; 3]>,
    options: &'a BRepBuildOptions,
    skipped_faces: &'a mut usize,
    skipped_edges: &'a mut usize,
}

fn resolve_face_surface(
    face_data: &topology::StepFace,
    entities: &EntityIndex,
    options: &BRepBuildOptions,
    skipped_faces: &mut usize,
) -> Option<SurfaceGeom> {
    if let Some(sid) = face_data.surface_id {
        if let Some(surface) = build_surface(sid, entities) {
            return Some(surface);
        }
        if options.allow_geometry_fallback {
            let surf_name = entities.get(&sid).map(|r| r.name.as_str()).unwrap_or("?");
            log::warn!(
                "[BRep] build_surface failed for #{} ({}), falling back to Plane from face edges",
                sid, surf_name
            );
            Some(fallback_plane_from_face(face_data))
        } else {
            *skipped_faces += 1;
            None
        }
    } else if options.allow_geometry_fallback {
        log::warn!("[BRep] face #{:?} has no surface reference, falling back to Plane from face edges",
            face_data.face_id);
        Some(fallback_plane_from_face(face_data))
    } else {
        *skipped_faces += 1;
        None
    }
}

/// Build a fallback plane from face edge vertices when the original surface cannot be resolved.
/// Uses Newell's method for an approximate normal and the vertex centroid as origin.
fn fallback_plane_from_face(face_data: &topology::StepFace) -> SurfaceGeom {
    let mut points: Vec<Vec3> = Vec::new();
    for bloop in &face_data.bounds {
        for edge in &bloop.edges {
            points.push(edge.start);
            points.push(edge.end);
        }
    }
    if points.is_empty() {
        return SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X };
    }

    // Centroid
    let inv_n = 1.0 / points.len() as f32;
    let origin = Vec3::new(
        points.iter().map(|p| p.x).sum::<f32>() * inv_n,
        points.iter().map(|p| p.y).sum::<f32>() * inv_n,
        points.iter().map(|p| p.z).sum::<f32>() * inv_n,
    );

    // Approximate normal via Newell's method (robust for non-planar polygons)
    let mut normal = Vec3::ZERO;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        normal.x += (points[i].y - points[j].y) * (points[i].z + points[j].z);
        normal.y += (points[i].z - points[j].z) * (points[i].x + points[j].x);
        normal.z += (points[i].x - points[j].x) * (points[i].y + points[j].y);
    }
    let normal_len = normal.length();
    let normal = if normal_len > 1e-10 { normal / normal_len } else { Vec3::Z };

    // u_dir from first edge direction
    let u_dir = if points.len() >= 2 {
        let d = points[1] - points[0];
        let dl = d.length();
        if dl > 1e-10 { d / dl } else { Vec3::X }
    } else {
        Vec3::X
    };
    let (u_dir, _v_dir) = plane_tangent_basis(normal, u_dir);

    SurfaceGeom::Plane { origin, normal, u_dir }
}

fn resolve_edge_curve(
    edge_data: &topology::StepEdge,
    entities: &EntityIndex,
    options: &BRepBuildOptions,
    skipped_edges: &mut usize,
) -> Option<CurveGeom> {
    if let Some(curve) = build_curve(edge_data.curve_id, entities) {
        return Some(curve);
    }
    if options.allow_geometry_fallback {
        let dir = edge_data.end - edge_data.start;
        let d = if dir.length() > 1e-10 { dir } else { Vec3::X };
        Some(CurveGeom::Line {
            origin: edge_data.start,
            direction: d,
        })
    } else {
        *skipped_edges += 1;
        None
    }
}

pub fn build_brep_with_options(
    entities: &EntityIndex,
    options: &BRepBuildOptions,
) -> Result<BRepBuildResult, StepError> {
    let mut reg = BRepRegistry::new();

    let solid_models = topology::collect_solid_models(entities);
    if solid_models.is_empty() {
        return Err(StepError::NoGeometry);
    }

    let tol = topology::global_tolerance(entities);
    let face_colors = topology::collect_face_colors(entities);
    let mut skipped_faces = 0usize;
    let mut skipped_edges = 0usize;
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
        },
    })
}
