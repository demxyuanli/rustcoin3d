use rc3d_core::math::Vec3;

use crate::geom::SurfaceGeom;
use crate::store::BRepRegistry;
use super::face_uv::{FaceUvLoops, uv_loop_is_degenerate};

/// True when UV bounds cover most of the native period (untrimmed analytic sheet).
pub fn uv_bounds_span_untrimmed_period(
    surface: &SurfaceGeom,
    bounds: (f32, f32, f32, f32),
) -> bool {
    let pr = surface.param_range();
    let du = (bounds.1 - bounds.0) / (pr.u_max - pr.u_min).max(1e-6);
    let dv = (bounds.3 - bounds.2) / (pr.v_max - pr.v_min).max(1e-6);
    du > 0.85 && dv > 0.85
}

/// Revolution trim fallback: native U from loop UV and V from axis-angle when boundary V is degenerate.
pub fn revolution_fallback_uv_bounds(
    loops: &FaceUvLoops,
    face: &crate::topo::BRepFace,
    global_vertices: &[Vec3],
) -> Option<(f32, f32, f32, f32)> {
    let (u0, u1, v0, v1) = loops.native_uv_bounds()?;
    if (u1 - u0).abs() < 1e-6 {
        return None;
    }
    if (v1 - v0).abs() >= 1e-5 {
        return Some((u0, u1, v0, v1));
    }
    let (rv0, rv1) = loops.revolution_v_bounds_from_3d(face, global_vertices)?;
    Some((u0, u1, rv0, rv1))
}

pub fn allows_trimmed_uv_grid(
    reg: &BRepRegistry,
    face: &crate::topo::BRepFace,
    loops: &FaceUvLoops,
    global_vertices: &[Vec3],
) -> bool {
    if loops.native_uv_bounds().is_none()
        && loops.uv_bounds_from_projection(face, global_vertices).is_none()
    {
        return false;
    }
    let wire_len = reg
        .wires
        .get(face.outer_wire)
        .map(|w| w.edges.len())
        .unwrap_or(0);
    if wire_len == 0 {
        return false;
    }
    if matches!(&face.surface, SurfaceGeom::BSpline(_)) && uv_loop_is_degenerate(loops) {
        return false;
    }
    matches!(
        &face.surface,
        SurfaceGeom::Revolution { .. }
            | SurfaceGeom::Cone { .. }
            | SurfaceGeom::Cylinder { .. }
            | SurfaceGeom::Sphere { .. }
            | SurfaceGeom::Torus { .. }
            | SurfaceGeom::Extrusion { .. }
            | SurfaceGeom::Offset { .. }
            | SurfaceGeom::BSpline(_)
            | SurfaceGeom::Plane { .. }
    )
}

pub fn allows_parametric_grid_fallback(
    reg: &BRepRegistry,
    face: &crate::topo::BRepFace,
    loops: &FaceUvLoops,
) -> bool {
    let Some(bounds) = loops.native_uv_bounds() else {
        return false;
    };
    let wire_len = reg
        .wires
        .get(face.outer_wire)
        .map(|w| w.edges.len())
        .unwrap_or(0);
    match &face.surface {
        SurfaceGeom::Revolution { .. }
        | SurfaceGeom::Cone { .. }
        | SurfaceGeom::Cylinder { .. }
        | SurfaceGeom::Offset { .. } => {
            if wire_len < 3 {
                return false;
            }
            !uv_bounds_span_untrimmed_period(&face.surface, bounds)
        }
        SurfaceGeom::BSpline(_) => false,
        _ => true,
    }
}

/// VERTEX_LOOP on a closed analytic surface: wire is only pole degeneracy (+ optional seam).
pub fn uses_closed_parametric_mesh(reg: &BRepRegistry, face: &crate::topo::BRepFace) -> bool {
    if !face.inner_wires.is_empty() {
        return false;
    }
    if !matches!(
        &face.surface,
        SurfaceGeom::Sphere { .. } | SurfaceGeom::Torus { .. }
    ) {
        return false;
    }
    let wire = match reg.wires.get(face.outer_wire) {
        Some(w) => w,
        None => return false,
    };
    if wire.edges.is_empty() {
        return true;
    }
    if face
        .degenerated_edges
        .iter()
        .any(|&dek| wire.edges.iter().any(|&(ek, _)| ek == dek))
    {
        return true;
    }
    wire.edges.iter().all(|&(ek, _)| {
        let Some(e) = reg.edges.get(ek) else {
            return false;
        };
        e.v_low == e.v_high || face.seam_edges.contains(&ek)
    })
}
