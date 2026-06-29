//! Location-based edge/vertex sharing (OCC TopLoc_Location equivalent).
//!
//! Post-processing pass that detects edges with identical curve geometry at
//! different spatial positions and assigns affine Location transforms so they
//! can share a canonical TShape.  This reduces V/E counts to match OCC's
//! reference output.

use rc3d_core::math::{Real, PVec3};
use crate::store::BRepStore;
use crate::topo::{EdgeKey, FaceKey, VertexKey, WireKey};
use crate::geom::CurveGeom;
use std::collections::HashMap;

/// Maximum edges to process for pairwise matching (O(n²) scan).
const MAX_EDGES_FOR_LOCATION_SCAN: usize = 256;

/// Run location-assignment pass. Returns the number of locations added.
pub fn assign_locations(reg: &mut BRepStore) -> usize {
    // Collect edges that are on periodic surfaces (cylinder, cone, sphere,
    // torus).  Location-based sharing only makes sense when the same curve
    // appears at different parameter positions on the same surface type.
    let edge_data: Vec<(EdgeKey, CurveGeom, VertexKey, VertexKey)> =
        reg.edges.iter()
            .filter(|(ek, _)| edge_on_periodic_face(reg, *ek))
            .map(|(ek, e)| (ek, e.curve.clone(), e.v_low, e.v_high))
            .collect();
    if edge_data.len() < 2 || edge_data.len() > MAX_EDGES_FOR_LOCATION_SCAN {
        return 0;
    }

    // ── Group edges by geometric signature ──────────────────────────
    let mut groups: HashMap<CurveSignature, Vec<usize>> = HashMap::new();
    for (i, (_ek, curve, _vl, _vh)) in edge_data.iter().enumerate() {
        if let Some(sig) = curve_signature(curve) {
            groups.entry(sig).or_default().push(i);
        }
    }

    let mut added = 0usize;

    for (_sig, indices) in &groups {
        if indices.len() < 2 {
            continue;
        }
        added += process_edge_group(reg, &edge_data, indices);
    }

    // Merge vertices that were canonicalized to the same position via location
    // transforms.  Only weld vertices with non-zero location indices — these
    // represent duplicate vertices created by location canonicalization.
    if added > 0 {
        // eprintln!("[loc] vertex_locations map: {:?}", reg.vertex_locations);
        for (&vk, &loc) in &reg.vertex_locations {
            if let Some(v) = reg.vertices.get(vk) {
                // eprintln!("[loc]   vk {:?} pos={:?} loc={}", vk, v.position, loc);
            }
        }
        let welded = weld_location_vertices(reg);
        // eprintln!("[loc] welded {} vertices", welded);
        if welded > 0 {
            added += welded;
        }
    }

    added
}

/// Check whether an edge belongs to at least one face with a periodic surface
/// (cylinder, cone, sphere, torus).  Only these surfaces benefit from
/// location-based edge sharing.
fn edge_on_periodic_face(reg: &BRepStore, ek: EdgeKey) -> bool {
    if let Some(face_keys) = reg.edge_to_faces.get(&ek) {
        face_keys.iter().any(|&fk| {
            reg.faces.get(fk).map_or(false, |f| {
                matches!(f.surface,
                    crate::geom::SurfaceGeom::Cylinder { .. } |
                    crate::geom::SurfaceGeom::Cone { .. } |
                    crate::geom::SurfaceGeom::Sphere { .. } |
                    crate::geom::SurfaceGeom::Torus { .. })
            })
        })
    } else {
        false
    }
}

// ── Curve signature for matching ────────────────────────────────────

/// The subset of curve parameters that must match for two edges to share
/// geometry via a rigid transform (rotation + translation only — no scaling).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CurveSignature {
    /// Circle: (radius, axis_direction).  Center varies.
    Circle { radius_bits: u64, axis_bits: [u64; 3] },
    /// Line: direction and chord length. (Origin varies.)
    Line { dir_bits: [u64; 3], len_bits: u64 },
}

fn float_bits(x: Real) -> u64 {
    x.to_bits()
}

fn pvec3_key(p: PVec3) -> [u64; 3] {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

/// Extract the underlying CurveGeom, unwrapping Trimmed to its basis.
fn unwrap_curve(curve: &CurveGeom) -> &CurveGeom {
    match curve {
        CurveGeom::Trimmed { basis, .. } => basis.as_ref(),
        other => other,
    }
}

fn curve_signature(curve: &CurveGeom) -> Option<CurveSignature> {
    let c = unwrap_curve(curve);
    match c {
        CurveGeom::Circle { radius, axis, .. } => {
            // Normalize axis direction for comparison
            let a = axis.normalize();
            let a_bits = [(-a.x).to_bits(), (-a.y).to_bits(), (-a.z).to_bits()];
            // Use epsilon-rounded radius to handle fp noise
            let r = (*radius * 1e6).round() / 1e6;
            Some(CurveSignature::Circle { radius_bits: float_bits(r), axis_bits: a_bits })
        }
        CurveGeom::Line { direction, .. } => {
            let len = direction.length();
            if len < 1e-12 {
                return None; // degenerated — skip
            }
            let dir = *direction / len;
            // Snap direction components to handle fp noise
            let snap = |x: Real| -> Real { (x * 1e6).round() / 1e6 };
            let sdir = PVec3::new(snap(dir.x), snap(dir.y), snap(dir.z)).normalize();
            let slen = (len * 1e6).round() / 1e6;
            Some(CurveSignature::Line {
                dir_bits: pvec3_key(sdir),
                len_bits: float_bits(slen),
            })
        }
        _ => None, // BSpline, Polyline, etc. — too complex for simple matching
    }
}

// ── Group processing ────────────────────────────────────────────────

fn process_edge_group(
    reg: &mut BRepStore,
    edge_data: &[(EdgeKey, CurveGeom, VertexKey, VertexKey)],
    indices: &[usize],
) -> usize {
    // Canonical = first edge in the group
    let canonical_idx = indices[0];
    let (canonical_ek, ref can_curve, can_v_lo, can_v_hi) = edge_data[canonical_idx];

    let mut added = 0usize;

    for &idx in &indices[1..] {
        let (ek, ref this_curve, this_v_lo, this_v_hi) = edge_data[idx];
        if ek == canonical_ek {
            continue;
        }

        // Compute rigid transform from canonical to this edge
        let loc = match compute_rigid_transform(unwrap_curve(can_curve), unwrap_curve(this_curve)) {
            Some(l) => l,
            None => continue,
        };

        // Verify: applying the inverse loc to this edge's vertices should
        // bring them close to the canonical vertices.
        let can_p_lo = vertex_pos(reg, can_v_lo);
        let can_p_hi = vertex_pos(reg, can_v_hi);
        let this_p_lo = vertex_pos(reg, this_v_lo);
        let this_p_hi = vertex_pos(reg, this_v_hi);

        // Apply inverse transform: p_canonical = loc⁻¹(p_this)
        let mut inv_p_lo = apply_inverse(&loc, this_p_lo);
        let mut inv_p_hi = apply_inverse(&loc, this_p_hi);

        let d_lo = (inv_p_lo - can_p_lo).length();
        let d_hi = (inv_p_hi - can_p_hi).length();
        let tol = 1e-3;

        // For self-loop edges (v_low == v_hi), verify the vertex is ON the
        // canonical curve, then snap to canonical vertex position so subsequent
        // vertex welding can merge them.
        let vertices_match = if can_v_lo == can_v_hi {
            let d = min_distance_to_curve(can_curve, inv_p_lo);
            if d < tol {
                // Snap to canonical vertex position for weld-ability
                inv_p_lo = can_p_lo;
                inv_p_hi = can_p_hi;
                true
            } else {
                false
            }
        } else {
            d_lo <= tol && d_hi <= tol
        };

        if !vertices_match {
            continue;
        }

        // ── Assign location ──────────────────────────────────────────
        let loc_idx = (reg.locations.len() + 1) as u8;

        if let Some(v) = reg.vertices.get_mut(this_v_lo) {
            v.position = inv_p_lo;
        }
        if this_v_lo != this_v_hi {
            if let Some(v) = reg.vertices.get_mut(this_v_hi) {
                v.position = inv_p_hi;
            }
        }

        reg.vertex_locations.insert(this_v_lo, loc_idx);
        if this_v_lo != this_v_hi {
            reg.vertex_locations.insert(this_v_hi, loc_idx);
        }

        reg.locations.push(loc);
        added += 1;
    }

    added
}

fn vertex_pos(reg: &BRepStore, vk: VertexKey) -> PVec3 {
    reg.vertices.get(vk).map(|v| v.position).unwrap_or(PVec3::ZERO)
}

// ── Transform helpers ───────────────────────────────────────────────

/// Compute the 3×4 affine matrix that maps points from canonical curve to
/// `other` curve.  Only handles rigid transforms (rotation + translation).
/// Returns row-major [r11,r12,r13,t1, r21,r22,r23,t2, r31,r32,r33,t3].
fn compute_rigid_transform(canonical: &CurveGeom, other: &CurveGeom) -> Option<[Real; 12]> {
    match (canonical, other) {
        (CurveGeom::Circle { center: c0, axis: a0, x_dir: x0, y_dir: y0, .. },
         CurveGeom::Circle { center: c1, axis: a1, x_dir: x1, y_dir: y1, .. }) =>
        {
            compute_rigid_from_frames(*c0, *a0, *x0, *y0, *c1, *a1, *x1, *y1)
        }
        (CurveGeom::Line { origin: o0, direction: d0 },
         CurveGeom::Line { origin: o1, direction: d1 }) =>
        {
            let len0 = d0.length();
            let len1 = d1.length();
            if len0 < 1e-12 || len1 < 1e-12 {
                return None;
            }
            let dir0 = *d0 / len0;
            let dir1 = *d1 / len1;
            // Build a minimal frame: Z = direction, X/Y = arbitrary orthogonal
            let z0 = dir0;
            let z1 = dir1;
            if (z0 - z1).length() < 1e-12 {
                // Pure translation along same line
                Some(identity_with_translation(*o1 - *o0))
            } else {
                // Rotation + translation: build frames and compute
                let x0 = build_perp(z0);
                let y0 = z0.cross(x0);
                let x1 = build_perp(z1);
                let y1 = z1.cross(x1);
                compute_rigid_from_frames(*o0, z0, x0, y0, *o1, z1, x1, y1)
            }
        }
        _ => None,
    }
}

fn build_perp(v: PVec3) -> PVec3 {
    let ref_dir = if v.x.abs() < 0.9 { PVec3::X } else { PVec3::Y };
    v.cross(ref_dir).normalize()
}

fn identity_with_translation(t: PVec3) -> [Real; 12] {
    [1.0, 0.0, 0.0, t.x,
     0.0, 1.0, 0.0, t.y,
     0.0, 0.0, 1.0, t.z]
}

fn compute_rigid_from_frames(
    c0: PVec3, a0: PVec3, x0: PVec3, y0: PVec3,
    c1: PVec3, a1: PVec3, x1: PVec3, y1: PVec3,
) -> Option<[Real; 12]> {
    // Build orthonormal frames
    let n0 = [x0.normalize(), y0.normalize(), a0.normalize()];
    let n1 = [x1.normalize(), y1.normalize(), a1.normalize()];

    // Rotation R: n1 → n0 (R maps canonical basis to other basis)
    // R[i][j] = n0[i].dot(n1[j])
    let r = [
        n0[0].dot(n1[0]), n0[0].dot(n1[1]), n0[0].dot(n1[2]),
        n0[1].dot(n1[0]), n0[1].dot(n1[1]), n0[1].dot(n1[2]),
        n0[2].dot(n1[0]), n0[2].dot(n1[1]), n0[2].dot(n1[2]),
    ];

    // Translation: t = c0 - R*c1 (maps canonical point to other point)
    // Actually: other_point = R * canonical_point + t
    // So: c1 = R * c0 + t → t = c1 - R * c0
    let rc0 = PVec3::new(
        r[0] * c0.x + r[1] * c0.y + r[2] * c0.z,
        r[3] * c0.x + r[4] * c0.y + r[5] * c0.z,
        r[6] * c0.x + r[7] * c0.y + r[8] * c0.z,
    );
    let t = c1 - rc0;

    Some([r[0], r[1], r[2], t.x,
          r[3], r[4], r[5], t.y,
          r[6], r[7], r[8], t.z])
}

/// Apply inverse rigid transform: p_canonical = Rᵀ·(p_other - t)
fn apply_inverse(loc: &[Real; 12], p: PVec3) -> PVec3 {
    // loc = [r11,r12,r13,t1, r21,r22,r23,t2, r31,r32,r33,t3]
    // Transform: p_other = R * p_canonical + t
    // Inverse: p_canonical = Rᵀ * (p_other - t)
    let px = p.x - loc[3];
    let py = p.y - loc[7];
    let pz = p.z - loc[11];
    // Rᵀ applied to (px, py, pz)
    PVec3::new(
        loc[0] * px + loc[4] * py + loc[8] * pz,
        loc[1] * px + loc[5] * py + loc[9] * pz,
        loc[2] * px + loc[6] * py + loc[10] * pz,
    )
}

/// Minimum distance from point to curve, sampled at 64 points.
fn min_distance_to_curve(curve: &CurveGeom, pt: PVec3) -> Real {
    let c = unwrap_curve(curve);
    let n = 64;
    let mut min_d = Real::MAX;
    for i in 0..=n {
        let t = i as Real / n as Real;
        let p = c.d0(t);
        let d = (p - pt).length();
        if d < min_d { min_d = d; }
    }
    min_d
}

// ── Vertex merging after location canonicalization ──────────────────

/// Weld vertices that have location transforms assigned and are now at the
/// same canonical position.  Only affects vertices with non-zero location
/// indices — avoids false merges on non-canonicalized vertices.
fn weld_location_vertices(reg: &mut BRepStore) -> usize {
    // Collect vertices with location indices
    let located: Vec<(VertexKey, u8)> = reg.vertex_locations.iter()
        .map(|(&vk, &loc)| (vk, loc))
        .collect();
    if located.len() < 2 { return 0; }

    // Build position groups for located vertices only
    let mut groups: HashMap<[u64; 3], Vec<VertexKey>> = HashMap::new();
    for &(vk, _loc) in &located {
        if let Some(v) = reg.vertices.get(vk) {
            let round = |x: Real| -> u64 { (x * 1e6).round().to_bits() };
            let key = [round(v.position.x), round(v.position.y), round(v.position.z)];
            groups.entry(key).or_default().push(vk);
        }
    }

    let mut removed = 0usize;
    let mut remap: HashMap<VertexKey, VertexKey> = HashMap::new();

    for eks in groups.values() {
        if eks.len() < 2 { continue; }
        let canonical = eks[0];
        for &old_vk in &eks[1..] {
            remap.insert(old_vk, canonical);
            removed += 1;
        }
    }

    if removed == 0 { return 0; }

    // Remap edge references
    for edge in reg.edges.values_mut() {
        if let Some(&nvk) = remap.get(&edge.v_low) { edge.v_low = nvk; }
        if let Some(&nvk) = remap.get(&edge.v_high) { edge.v_high = nvk; }
    }

    // Remap edge_hash_index
    let old_hash: Vec<((VertexKey, VertexKey), Vec<EdgeKey>)> =
        reg.edge_hash_index.drain().collect();
    for ((v0, v1), eks) in old_hash {
        let nv0 = remap.get(&v0).copied().unwrap_or(v0);
        let nv1 = remap.get(&v1).copied().unwrap_or(v1);
        let nk = if nv0 < nv1 { (nv0, nv1) } else { (nv1, nv0) };
        let entry = reg.edge_hash_index.entry(nk).or_default();
        for ek in eks {
            if !entry.contains(&ek) { entry.push(ek); }
        }
    }

    // Remap vertex_to_edges
    let old_v2e: Vec<(VertexKey, Vec<EdgeKey>)> =
        reg.vertex_to_edges.drain().collect();
    for (vk, eks) in old_v2e {
        let nvk = remap.get(&vk).copied().unwrap_or(vk);
        let entry = reg.vertex_to_edges.entry(nvk).or_default();
        for ek in eks {
            if !entry.contains(&ek) { entry.push(ek); }
        }
    }

    // Remove old vertices from SlotMap
    for old_vk in remap.keys() {
        reg.vertices.remove(*old_vk);
    }

    removed
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::BRepStore;
    use crate::topo::*;
    use crate::geom::CurveGeom;

    #[test]
    fn assign_locations_for_circles_at_different_z() {
        let mut reg = BRepStore::new();
        let v0 = reg.vertices.insert(BRepVertex { position: PVec3::new(5.0, 0.0, 0.0), tolerance: 1e-4 });
        let v1 = reg.vertices.insert(BRepVertex { position: PVec3::new(5.0, 0.0, 10.0), tolerance: 1e-4 });

        // Create two DIFFERENT cylindrical faces so the edges can be merged
        // via location sharing (edges_share_face prevents merging on same face).
        let face_a = reg.faces.insert(BRepFace {
            surface: crate::geom::SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0),
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });
        let face_b = reg.faces.insert(BRepFace {
            surface: crate::geom::SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 5.0),
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Circle at z=0
        let ek0 = crate::geom::CurveGeom::Circle {
            center: PVec3::new(0.0, 0.0, 0.0),
            axis: PVec3::Z,
            radius: 5.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        // Circle at z=10 — same radius, just translated
        let ek1 = crate::geom::CurveGeom::Circle {
            center: PVec3::new(0.0, 0.0, 10.0),
            axis: PVec3::Z,
            radius: 5.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };

        reg.add_edge_with_pcurve(v0, v0, ek0, 1e-4, face_a, crate::geom::curve2d::Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) }, true);
        reg.add_edge_with_pcurve(v1, v1, ek1, 1e-4, face_b, crate::geom::curve2d::Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) }, true);

        let n = assign_locations(&mut reg);
        assert_eq!(n, 1, "one location added");
        assert_eq!(reg.locations.len(), 1);
        // Vertex at z=10 should now be at canonical z=0 position
        let v1_pos = reg.vertices.get(v1).unwrap().position;
        assert!((v1_pos.z).abs() < 1e-3, "vertex should be at canonical z≈0, got {}", v1_pos.z);
        assert!(reg.vertex_locations.contains_key(&v1));
    }
}
