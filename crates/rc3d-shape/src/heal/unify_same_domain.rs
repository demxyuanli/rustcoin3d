//! Unify adjacent faces that share the same geometry into single faces.
//!
//! OCC alignment: ShapeUpgrade_UnifySameDomain::Perform()
//!
//! Algorithm:
//! 1. Build face adjacency map from shared edges
//! 2. For each adjacent pair, check if surfaces are geometrically identical
//! 3. Union-find to group mergeable faces into clusters
//! 4. For each cluster of 2+ faces, merge into a single face
//! 5. Update shell with new faces

use crate::store::BRepStore;
use crate::topo::*;
use crate::geom::SurfaceGeom;
use crate::geom::curve2d::Curve2d;
use crate::nurbs::NurbsSurface;
use super::wire_ops::reorder_wire_edges;
use rc3d_core::math::Real;
use std::collections::{HashMap, HashSet};

// ── Result types ──────────────────────────────────────────────────────

/// Merge result: maps old face keys to their new unified face.
#[derive(Debug, Clone)]
pub struct UnifyResult {
    pub merge_map: HashMap<FaceKey, FaceKey>,
    pub merges: usize,
}

impl UnifyResult {
    pub fn empty() -> Self {
        Self {
            merge_map: HashMap::new(),
            merges: 0,
        }
    }
}

// ── Union-find for face clustering ────────────────────────────────────

struct UnionFind {
    parent: HashMap<FaceKey, FaceKey>,
}

impl UnionFind {
    fn new() -> Self {
        Self { parent: HashMap::new() }
    }

    fn make_set(&mut self, x: FaceKey) {
        self.parent.entry(x).or_insert(x);
    }

    fn find(&mut self, x: FaceKey) -> FaceKey {
        let p = *self.parent.get(&x).unwrap_or(&x);
        if p != x {
            let root = self.find(p);
            self.parent.insert(x, root);
        }
        *self.parent.get(&x).unwrap_or(&x)
    }

    fn union(&mut self, x: FaceKey, y: FaceKey) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent.insert(ry, rx);
        }
    }

    fn clusters(&mut self) -> HashMap<FaceKey, Vec<FaceKey>> {
        let mut groups: HashMap<FaceKey, Vec<FaceKey>> = HashMap::new();
        let keys: Vec<FaceKey> = self.parent.keys().copied().collect();
        for k in keys {
            let root = self.find(k);
            groups.entry(root).or_default().push(k);
        }
        groups
    }
}

// ── Surface comparison ────────────────────────────────────────────────

/// Check if two BSpline surfaces are exactly the same.
fn same_bspline(a: &NurbsSurface, b: &NurbsSurface) -> bool {
    if a.degree_u != b.degree_u || a.degree_v != b.degree_v {
        return false;
    }
    if a.control_points.len() != b.control_points.len() {
        return false;
    }
    // Compare control points
    for (row_a, row_b) in a.control_points.iter().zip(b.control_points.iter()) {
        if row_a.len() != row_b.len() {
            return false;
        }
        for (cp_a, cp_b) in row_a.iter().zip(row_b.iter()) {
            if (cp_a - *cp_b).length_squared() > 1e-12 {
                return false;
            }
        }
    }
    // Compare weights
    for (row_a, row_b) in a.weights.iter().zip(b.weights.iter()) {
        if row_a.len() != row_b.len() {
            return false;
        }
        for (w_a, w_b) in row_a.iter().zip(row_b.iter()) {
            if (w_a - w_b).abs() > 1e-12 {
                return false;
            }
        }
    }
    // Compare knots
    if a.knots_u.len() != b.knots_u.len() || a.knots_v.len() != b.knots_v.len() {
        return false;
    }
    for (k_a, k_b) in a.knots_u.iter().zip(b.knots_u.iter()) {
        if (k_a - k_b).abs() > 1e-12 {
            return false;
        }
    }
    for (k_a, k_b) in a.knots_v.iter().zip(b.knots_v.iter()) {
        if (k_a - k_b).abs() > 1e-12 {
            return false;
        }
    }
    true
}

/// Check if two surfaces represent the same geometry within tolerance.
pub fn same_surface(a: &SurfaceGeom, b: &SurfaceGeom, tol: Real) -> bool {
    use SurfaceGeom::*;
    match (a, b) {
        (Plane { origin: o_a, normal: n_a, u_dir: u_a },
         Plane { origin: o_b, normal: n_b, u_dir: u_b }) => {
            // Normal must match (within tolerance)
            if (n_a - *n_b).length() > tol && (n_a + *n_b).length() > tol {
                return false;
            }
            // U-direction must match (within tolerance)
            if (u_a - *u_b).length() > tol && (u_a + *u_b).length() > tol {
                return false;
            }
            // Origin distance along normal must match
            let d_a = n_a.dot(*o_a);
            let d_b = n_b.dot(*o_b);
            (d_a - d_b).abs() <= tol
        }

        (Cylinder { origin: o_a, axis: ax_a, radius: r_a, x_dir: _xda, y_dir: _yda },
         Cylinder { origin: o_b, axis: ax_b, radius: r_b, x_dir: _xdb, y_dir: _ydb }) => {
            if (r_a - r_b).abs() > tol {
                return false;
            }
            // Axis must be collinear
            let ax_same = (ax_a - *ax_b).length() <= tol || (ax_a + *ax_b).length() <= tol;
            if !ax_same {
                return false;
            }
            // Origin must lie on same line along axis
            // Project o_a onto ax_b and check distance to o_b along axis
            let diff = *o_a - *o_b;
            let along_axis = diff.dot(*ax_b) * *ax_b;
            let perp_dist = (diff - along_axis).length();
            perp_dist <= tol
        }

        (Cone { apex: ap_a, axis: ax_a, semi_angle: sa_a, radius_at_apex: ra_a, x_dir: _xda, y_dir: _yda },
         Cone { apex: ap_b, axis: ax_b, semi_angle: sa_b, radius_at_apex: ra_b, x_dir: _xdb, y_dir: _ydb }) => {
            if (sa_a - sa_b).abs() > tol {
                return false;
            }
            if (ra_a - ra_b).abs() > tol {
                return false;
            }
            let ax_same = (ax_a - *ax_b).length() <= tol || (ax_a + *ax_b).length() <= tol;
            if !ax_same {
                return false;
            }
            // Apex must lie on the same axis line
            let diff = *ap_a - *ap_b;
            let along_axis = diff.dot(*ax_b) * *ax_b;
            let perp_dist = (diff - along_axis).length();
            perp_dist <= tol
        }

        (Sphere { center: c_a, radius: r_a },
         Sphere { center: c_b, radius: r_b }) => {
            (c_a - *c_b).length() <= tol && (r_a - r_b).abs() <= tol
        }

        (Torus { center: c_a, axis: ax_a, major_r: mr_a, minor_r: mnr_a, x_dir: _xda, y_dir: _yda },
         Torus { center: c_b, axis: ax_b, major_r: mr_b, minor_r: mnr_b, x_dir: _xdb, y_dir: _ydb }) => {
            if (mr_a - mr_b).abs() > tol || (mnr_a - mnr_b).abs() > tol {
                return false;
            }
            let ax_same = (ax_a - *ax_b).length() <= tol || (ax_a + *ax_b).length() <= tol;
            if !ax_same {
                return false;
            }
            // Center must lie on the same axis line
            let diff = *c_a - *c_b;
            let along_axis = diff.dot(*ax_b) * *ax_b;
            let perp_dist = (diff - along_axis).length();
            perp_dist <= tol
        }

        (BSpline(n_a), BSpline(n_b)) => same_bspline(n_a, n_b),

        // Extrusion / Revolution / Offset: fallback to tolerant comparison
        // Sample points and check if distances are within tolerance
        (Extrusion { generatrix: ga, direction: da },
         Extrusion { generatrix: gb, direction: db }) => {
            if (da - *db).length() > tol && (da + *db).length() > tol {
                return false;
            }
            // Compare generatrix curves by sampling
            let n = 8;
            for i in 0..=n {
                let t = i as Real / n as Real;
                if (ga.d0(t) - gb.d0(t)).length() > tol {
                    return false;
                }
            }
            true
        }

        (Revolution { generatrix: ga, axis_origin: ao_a, axis_dir: ad_a },
         Revolution { generatrix: gb, axis_origin: ao_b, axis_dir: ad_b }) => {
            if (ad_a - *ad_b).length() > tol && (ad_a + *ad_b).length() > tol {
                return false;
            }
            if (ao_a - *ao_b).length() > tol {
                return false;
            }
            // Compare generatrix
            let n = 8;
            for i in 0..=n {
                let t = i as Real / n as Real;
                if (ga.d0(t) - gb.d0(t)).length() > tol {
                    return false;
                }
            }
            true
        }

        (Offset { basis: ba, distance: da },
         Offset { basis: bb, distance: db }) => {
            (da - db).abs() <= tol && same_surface(ba, bb, tol)
        }

        // Different variants: not the same surface
        _ => false,
    }
}

// ── Face edge collection ──────────────────────────────────────────────

/// Get all edge keys from all wires of a face.
fn all_face_edges(face_key: FaceKey, reg: &BRepStore) -> Vec<(EdgeKey, Orientation)> {
    let face = match reg.faces.get(face_key) {
        Some(f) => f,
        None => return vec![],
    };

    let mut result = Vec::new();
    let mut wires_to_collect = vec![face.outer_wire];
    wires_to_collect.extend(&face.inner_wires);

    for &wk in &wires_to_collect {
        if let Some(wire) = reg.wires.get(wk) {
            result.extend(wire.edges.iter().copied());
        }
    }

    result
}

// ── Face merging ──────────────────────────────────────────────────────

/// Merge two adjacent faces that share the same geometry into one face.
/// Returns the new face key and a list of edges that were removed (shared edges).
pub(crate) fn merge_face_pair(
    face_a: FaceKey,
    face_b: FaceKey,
    reg: &mut BRepStore,
) -> Option<(FaceKey, Vec<EdgeKey>)> {
    let face_data_a = reg.faces.get(face_a)?.clone();
    let face_data_b = reg.faces.get(face_b)?;

    // Collect all edges from both faces
    let edges_a = all_face_edges(face_a, reg);
    let edges_b = all_face_edges(face_b, reg);

    // Identify shared edges between the two faces
    let shared: HashSet<EdgeKey> = {
        let set_a: HashSet<EdgeKey> = edges_a.iter().map(|(ek, _)| *ek).collect();
        let set_b: HashSet<EdgeKey> = edges_b.iter().map(|(ek, _)| *ek).collect();
        set_a.intersection(&set_b).copied().collect()
    };

    if shared.is_empty() {
        // Not adjacent via shared edges — cannot merge
        return None;
    }

    // Construct merged outer wire: all outer edges from both faces minus shared edges
    let outer_a: Vec<(EdgeKey, Orientation)> = {
        let wire = reg.wires.get(face_data_a.outer_wire)?;
        wire.edges.clone()
    };
    let outer_b: Vec<(EdgeKey, Orientation)> = {
        let wire = reg.wires.get(face_data_b.outer_wire)?;
        wire.edges.clone()
    };

    let mut merged_outer: Vec<(EdgeKey, Orientation)> = Vec::new();
    let removed: Vec<EdgeKey> = shared.iter().copied().collect();

    for (ek, orient) in outer_a {
        if !shared.contains(&ek) {
            merged_outer.push((ek, orient));
        }
    }
    for (ek, orient) in outer_b {
        if !shared.contains(&ek) {
            merged_outer.push((ek, orient));
        }
    }

    if merged_outer.is_empty() {
        return None;
    }

    // Reorder the merged edges into a connected loop
    let reordered = reorder_wire_edges(&merged_outer, reg)?;

    // Collect inner wires from both faces
    let mut inner_wires = face_data_a.inner_wires.clone();
    inner_wires.extend(face_data_b.inner_wires.clone());

    // Collect seam edges and degenerated edges (deduplicated)
    let mut seam_edges = face_data_a.seam_edges.clone();
    for ek in &face_data_b.seam_edges {
        if !seam_edges.contains(ek) {
            seam_edges.push(*ek);
        }
    }
    let mut degenerated_edges = face_data_a.degenerated_edges.clone();
    for ek in &face_data_b.degenerated_edges {
        if !degenerated_edges.contains(ek) {
            degenerated_edges.push(*ek);
        }
    }

    // Create new outer wire
    let new_outer_wire = reg.wires.insert(BRepWire { edges: reordered });

    // Use surface from face_a (already verified same as face_b)
    let new_face = BRepFace {
        surface: face_data_a.surface.clone(),
        outer_wire: new_outer_wire,
        inner_wires,
        same_sense: face_data_a.same_sense,
        tolerance: face_data_a.tolerance.max(face_data_b.tolerance),
        seam_edges,
        color: face_data_a.color.or(face_data_b.color),
        degenerated_edges,
    };

    let new_fk = reg.faces.insert(new_face);

    // Update edge_to_faces index for all outer edges of the new face
    if let Some(wire) = reg.wires.get(new_outer_wire) {
        for &(ek, _) in &wire.edges {
            let entry = reg.edge_to_faces.entry(ek).or_default();
            if !entry.contains(&new_fk) {
                entry.push(new_fk);
            }
        }
    }

    // Transfer PCurves from old faces to new face for the remaining outer edges
    // and for inner wire edges
    for wk in std::iter::once(&new_outer_wire).chain(reg.faces.get(new_fk)?.inner_wires.iter()) {
        let wire = reg.wires.get(*wk)?;
        for &(ek, _) in &wire.edges {
            // Collect the PCurve to copy before borrowing reg.edges mutably
            let pc_to_copy: Option<Curve2d> = reg.edges.get(ek).and_then(|edge| {
                edge.pcurves.get(&face_a)
                    .or_else(|| edge.pcurves.get(&face_b))
                    .cloned()
            });
            if let Some(pc) = pc_to_copy {
                if let Some(edge_mut) = reg.edges.get_mut(ek) {
                    edge_mut.pcurves.entry(new_fk).or_insert(pc);
                }
            }
        }
    }

    Some((new_fk, removed))
}

/// Merge a cluster of faces (2+) into a single face by iteratively merging pairs.
fn merge_face_cluster(
    cluster: &[FaceKey],
    reg: &mut BRepStore,
) -> Option<(FaceKey, HashMap<FaceKey, FaceKey>)> {
    if cluster.is_empty() {
        return None;
    }
    if cluster.len() == 1 {
        let mut map = HashMap::new();
        map.insert(cluster[0], cluster[0]);
        return Some((cluster[0], map));
    }

    let mut merge_map: HashMap<FaceKey, FaceKey> = HashMap::new();
    let mut current = cluster[0];

    // Mark all faces as mapping to themselves initially
    for &fk in cluster {
        merge_map.insert(fk, fk);
    }

    // Iteratively merge adjacent pairs
    for &next_fk in &cluster[1..] {
        // Check if current and next_fk are still valid in the store
        if reg.faces.get(current).is_none() || reg.faces.get(next_fk).is_none() {
            continue;
        }

        // Check adjacency via shared edges
        let shared = reg.find_shared_edges(current, next_fk);
        if shared.is_empty() {
            // Faces are in same cluster (union-find group) but may not be
            // directly adjacent — they got grouped transitively.
            // Skip for now; all faces in cluster share the same surface.
            continue;
        }

        if let Some((merged_fk, _removed)) = merge_face_pair(current, next_fk, reg) {
            // Update mapping: both old faces map to the merged face
            for fk in cluster {
                let current_target = merge_map.get(fk).copied().unwrap_or(*fk);
                if current_target == current || current_target == next_fk {
                    merge_map.insert(*fk, merged_fk);
                }
            }
            merge_map.insert(current, merged_fk);
            merge_map.insert(next_fk, merged_fk);
            current = merged_fk;
        }
    }

    // Update mapping for any face not yet assigned
    for &fk in cluster {
        merge_map.entry(fk).or_insert(current);
    }

    Some((current, merge_map))
}

// ── Main API ──────────────────────────────────────────────────────────

/// Unify faces with the same geometry in a shell.
///
/// Merges adjacent faces that share the same underlying surface geometry
/// (e.g., coplanar planes, coaxial cylinders) into single faces.
///
/// OCC alignment: ShapeUpgrade_UnifySameDomain::Perform()
pub fn unify_same_domain(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    tolerance: Real,
) -> UnifyResult {
    // 1. Collect faces from shell
    let face_keys: Vec<FaceKey> = {
        let shell = match reg.shells.get(shell_key) {
            Some(s) => s,
            None => return UnifyResult::empty(),
        };
        shell.faces.iter().map(|&(fk, _)| fk).collect()
    };

    if face_keys.len() < 2 {
        return UnifyResult::empty();
    }

    // 2. Build face adjacency map from shared edges
    let mut adjacency: HashMap<FaceKey, HashSet<FaceKey>> = HashMap::new();
    for &fk in &face_keys {
        adjacency.entry(fk).or_default();
    }

    // Use edge_to_faces index for efficient adjacency discovery
    for &fk in &face_keys {
        let face = match reg.faces.get(fk) {
            Some(f) => f,
            None => continue,
        };
        let mut visited_edges = HashSet::new();
        for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
            let wire = match reg.wires.get(*wire_key) {
                Some(w) => w,
                None => continue,
            };
            for &(ek, _) in &wire.edges {
                if !visited_edges.insert(ek) {
                    continue;
                }
                if let Some(adjacent_faces) = reg.edge_to_faces.get(&ek) {
                    for &other_fk in adjacent_faces {
                        if other_fk != fk && adjacency.contains_key(&other_fk) {
                            adjacency.get_mut(&fk).unwrap().insert(other_fk);
                        }
                    }
                }
            }
        }
    }

    // 3. Check same-surface condition for adjacent pairs, union-find to group
    let mut uf = UnionFind::new();
    for &fk in &face_keys {
        uf.make_set(fk);
    }

    for (&fk_a, neighbors) in &adjacency {
        let surf_a = match reg.faces.get(fk_a) {
            Some(f) => f.surface.clone(),
            None => continue,
        };
        for &fk_b in neighbors {
            if fk_b <= fk_a {
                continue; // Process each pair once
            }
            let surf_b = match reg.faces.get(fk_b) {
                Some(f) => &f.surface,
                None => continue,
            };
            if same_surface(&surf_a, surf_b, tolerance) {
                uf.union(fk_a, fk_b);
            }
        }
    }

    // 4. Group faces into merge clusters (only clusters with 2+ faces need merging)
    let clusters = uf.clusters();
    let mut merge_map: HashMap<FaceKey, FaceKey> = HashMap::new();
    let mut merges = 0usize;

    for (_, cluster) in &clusters {
        if cluster.len() < 2 {
            // Singleton: keep as-is
            if !cluster.is_empty() {
                merge_map.entry(cluster[0]).or_insert(cluster[0]);
            }
            continue;
        }

        if let Some((merged_fk, cluster_map)) = merge_face_cluster(cluster, reg) {
            for (old_fk, new_fk) in cluster_map {
                merge_map.insert(old_fk, new_fk);
            }
            if cluster.len() > 1 {
                merges += 1;
            }
            let _ = merged_fk; // merged face created
        } else {
            // Merge failed — keep original faces
            for &fk in cluster {
                merge_map.entry(fk).or_insert(fk);
            }
        }
    }

    // Ensure all original faces have a mapping
    for fk in &face_keys {
        merge_map.entry(*fk).or_insert(*fk);
    }

    // 5. Update shell: replace old faces with merged faces (deduplicated)
    {
        let shell = match reg.shells.get_mut(shell_key) {
            Some(s) => s,
            None => return UnifyResult { merge_map, merges },
        };

        let mut new_face_set: HashSet<FaceKey> = HashSet::new();
        let mut new_faces: Vec<(FaceKey, Orientation)> = Vec::new();

        for &(old_fk, orient) in &shell.faces {
            let resolved_fk = merge_map.get(&old_fk).copied().unwrap_or(old_fk);
            if new_face_set.insert(resolved_fk) {
                new_faces.push((resolved_fk, orient));
            }
        }

        shell.faces = new_faces;
    }

    UnifyResult { merge_map, merges }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::CurveGeom;
    use crate::geom::curve2d::Curve2d;
    use rc3d_core::math::PVec3;

    /// Build two adjacent coplanar rectangles sharing one edge.
    /// Returns (shell_key, left_face_key, right_face_key, shared_edge_key).
    fn build_coplanar_rectangles(reg: &mut BRepStore) -> (ShellKey, FaceKey, FaceKey, EdgeKey) {
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };

        // Left rectangle: (0,0)-(1,0)-(1,1)-(0,1)
        // Right rectangle: (1,0)-(2,0)-(2,1)-(1,1)
        // Shared edge: (1,0)-(1,1)

        let v00 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v10 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let v20 = reg.find_or_add_vertex(PVec3::new(2.0, 0.0, 0.0), 1e-4);
        let v01 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-4);
        let v11 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let v21 = reg.find_or_add_vertex(PVec3::new(2.0, 1.0, 0.0), 1e-4);

        let line = |a: PVec3, b: PVec3| CurveGeom::Line {
            origin: a,
            direction: b - a,
        };
        let pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };

        // Create faces first
        let f_left = reg.add_face(surface.clone(), 1e-4);
        let f_right = reg.add_face(surface.clone(), 1e-4);

        // Left face: bottom edge v00→v10, right edge v10→v11, top edge v11→v01, left edge v01→v00
        let e_bottom_l = reg.add_edge_with_pcurve(
            v00, v10, line(PVec3::new(0.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 0.0)),
            1e-4, f_left, pc(PVec3::new(0.0, 0.0, 0.0), PVec3::new(1.0, 0.0, 0.0)), true);
        let e_right_l = reg.add_edge_with_pcurve(
            v10, v11, line(PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
            1e-4, f_left, pc(PVec3::new(1.0, 0.0, 0.0), PVec3::new(1.0, 1.0, 0.0)), true);
        let e_top_l = reg.add_edge_with_pcurve(
            v11, v01, line(PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)),
            1e-4, f_left, pc(PVec3::new(1.0, 1.0, 0.0), PVec3::new(0.0, 1.0, 0.0)), true);
        let e_left_l = reg.add_edge_with_pcurve(
            v01, v00, line(PVec3::new(0.0, 1.0, 0.0), PVec3::new(0.0, 0.0, 0.0)),
            1e-4, f_left, pc(PVec3::new(0.0, 1.0, 0.0), PVec3::new(0.0, 0.0, 0.0)), true);

        // Right face: bottom edge v10→v20, right edge v20→v21, top edge v21→v11, left edge v11→v10
        // Reuse the shared edge (e_right_l) for the right face's left boundary,
        // but in reversed direction at the wire level.
        // Add pcurve for f_right on the existing shared edge.
        reg.set_pcurve(e_right_l, f_right,
            Curve2d::Line { origin: (1.0, 0.0), direction: (0.0, 1.0) }, true);
        let e_left_r = e_right_l; // Same edge key

        let e_bottom_r = reg.add_edge_with_pcurve(
            v10, v20, line(PVec3::new(1.0, 0.0, 0.0), PVec3::new(2.0, 0.0, 0.0)),
            1e-4, f_right, pc(PVec3::new(1.0, 0.0, 0.0), PVec3::new(2.0, 0.0, 0.0)), true);
        let e_right_r = reg.add_edge_with_pcurve(
            v20, v21, line(PVec3::new(2.0, 0.0, 0.0), PVec3::new(2.0, 1.0, 0.0)),
            1e-4, f_right, pc(PVec3::new(2.0, 0.0, 0.0), PVec3::new(2.0, 1.0, 0.0)), true);
        let e_top_r = reg.add_edge_with_pcurve(
            v21, v11, line(PVec3::new(2.0, 1.0, 0.0), PVec3::new(1.0, 1.0, 0.0)),
            1e-4, f_right, pc(PVec3::new(2.0, 1.0, 0.0), PVec3::new(1.0, 1.0, 0.0)), true);

        let shared_edge = e_right_l; // The edge from (1,0) to (1,1)

        // Set wires
        let orient_for = |ek: EdgeKey, from_vk: VertexKey| -> Orientation {
            let edge = reg.edges.get(ek).unwrap();
            if edge.v_low == from_vk {
                Orientation::Forward
            } else {
                Orientation::Reversed
            }
        };

        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_left).unwrap().outer_wire) {
            w.edges = vec![
                (e_bottom_l, orient_for(e_bottom_l, v00)),
                (e_right_l, orient_for(e_right_l, v10)),
                (e_top_l, orient_for(e_top_l, v11)),
                (e_left_l, orient_for(e_left_l, v01)),
            ];
        }
        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_right).unwrap().outer_wire) {
            w.edges = vec![
                (e_bottom_r, orient_for(e_bottom_r, v10)),
                (e_right_r, orient_for(e_right_r, v20)),
                (e_top_r, orient_for(e_top_r, v21)),
                (e_left_r, orient_for(e_left_r, v11)),
            ];
        }

        // Rebuild edge_to_faces index to include manually added pcurves
        reg.build_edge_to_faces_index();

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f_left, Orientation::Forward), (f_right, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        (sk, f_left, f_right, shared_edge)
    }

    #[test]
    fn test_same_surface_plane() {
        let a = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let b = SurfaceGeom::Plane {
            origin: PVec3::new(0.0, 0.0, 1e-7),
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        assert!(same_surface(&a, &b, 1e-6));

        let c = SurfaceGeom::Plane {
            origin: PVec3::new(0.0, 0.0, 1.0),
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        assert!(!same_surface(&a, &c, 1e-6));
    }

    #[test]
    fn test_same_surface_cylinder() {
        let s = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);
        let a = SurfaceGeom::Cylinder {
            origin: PVec3::ZERO,
            axis: PVec3::Z,
            radius: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        assert!(same_surface(&s, &a, 1e-6));

        let b = SurfaceGeom::Cylinder {
            origin: PVec3::new(0.0, 0.0, 5.0),
            axis: PVec3::Z,
            radius: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        assert!(same_surface(&a, &b, 1e-6), "same axis, same radius, different origin along axis");

        let c = SurfaceGeom::Cylinder {
            origin: PVec3::ZERO,
            axis: PVec3::Z,
            radius: 2.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        assert!(!same_surface(&a, &c, 1e-6), "different radius");
    }

    #[test]
    fn test_same_surface_sphere() {
        let a = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 1.0 };
        let b = SurfaceGeom::Sphere { center: PVec3::new(1e-7, 0.0, 0.0), radius: 1.0 };
        assert!(same_surface(&a, &b, 1e-6));

        let c = SurfaceGeom::Sphere { center: PVec3::new(1.0, 0.0, 0.0), radius: 1.0 };
        assert!(!same_surface(&a, &c, 1e-6));
    }

    #[test]
    fn test_same_surface_different_variants() {
        let plane = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let sphere = SurfaceGeom::Sphere { center: PVec3::ZERO, radius: 1.0 };
        assert!(!same_surface(&plane, &sphere, 1e-6));
    }

    #[test]
    fn test_merge_coplanar_rectangles() {
        let mut reg = BRepStore::new();
        let (sk, f_left, f_right, shared_edge) = build_coplanar_rectangles(&mut reg);

        let result = unify_same_domain(sk, &mut reg, 1e-6);

        assert_eq!(result.merges, 1, "should merge two coplanar rectangles into one");

        // Shell should have 1 face
        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 1, "shell should have exactly 1 face after merge");

        let merged_fk = shell.faces[0].0;
        let merged_face = reg.faces.get(merged_fk).unwrap();

        // Merged face should have 6 outer edges (4+4-2 = 6: bottom x2, top x2, left, right)
        let merged_wire = reg.wires.get(merged_face.outer_wire).unwrap();
        assert_eq!(merged_wire.edges.len(), 6,
            "merged outer wire should have 6 edges (4+4-2 shared)");

        // The shared edge should NOT be in the merged outer wire
        let edge_set: HashSet<EdgeKey> = merged_wire.edges.iter().map(|(ek, _)| *ek).collect();
        assert!(!edge_set.contains(&shared_edge),
            "shared edge should be removed from merged face");

        // Verify merge_map
        let merged_target = result.merge_map.get(&f_left).copied();
        assert!(merged_target.is_some());
        // Both original faces should map to the same merged face
        assert_eq!(
            result.merge_map.get(&f_left),
            result.merge_map.get(&f_right),
            "both faces should map to same merged face"
        );
    }

    #[test]
    fn test_no_merge_different_surfaces() {
        let mut reg = BRepStore::new();

        let plane_surf = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };
        let cyl_surf = SurfaceGeom::cylinder(PVec3::ZERO, PVec3::Z, 1.0);

        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-4);

        let f_plane = reg.add_face(plane_surf.clone(), 1e-4);
        let f_cyl = reg.add_face(cyl_surf.clone(), 1e-4);

        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };

        // Shared edge between the two faces
        let e_shared = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f_plane, pc.clone(), true);
        let _ = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f_cyl, pc.clone(), true);
        // Remaining edges for f_plane
        let e2 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, f_plane, pc.clone(), true);
        let e3 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, f_plane, pc.clone(), true);
        let e4 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, f_plane, pc.clone(), true);
        // Remaining edges for f_cyl
        let e5 = reg.add_edge_with_pcurve(v1, v2, line.clone(), 1e-4, f_cyl, pc.clone(), true);
        let e6 = reg.add_edge_with_pcurve(v2, v3, line.clone(), 1e-4, f_cyl, pc.clone(), true);
        let e7 = reg.add_edge_with_pcurve(v3, v0, line.clone(), 1e-4, f_cyl, pc.clone(), true);

        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_plane).unwrap().outer_wire) {
            w.edges = vec![
                (e_shared, Orientation::Forward),
                (e2, Orientation::Forward),
                (e3, Orientation::Forward),
                (e4, Orientation::Forward),
            ];
        }
        if let Some(w) = reg.wires.get_mut(reg.faces.get(f_cyl).unwrap().outer_wire) {
            w.edges = vec![
                (e_shared, Orientation::Reversed),
                (e5, Orientation::Forward),
                (e6, Orientation::Forward),
                (e7, Orientation::Forward),
            ];
        }

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(f_plane, Orientation::Forward), (f_cyl, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        let result = unify_same_domain(sk, &mut reg, 1e-6);
        assert_eq!(result.merges, 0, "different surfaces should not merge");
        let shell = reg.shells.get(sk).unwrap();
        assert_eq!(shell.faces.len(), 2, "shell should still have 2 faces");
    }

    #[test]
    fn test_same_surface_cone() {
        let a = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, 0.5, 0.0);
        let b = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, 0.5, 0.0);
        assert!(same_surface(&a, &b, 1e-6));

        let c = SurfaceGeom::cone(PVec3::ZERO, PVec3::Z, 0.3, 0.0);
        assert!(!same_surface(&a, &c, 1e-6));
    }

    #[test]
    fn test_same_surface_torus() {
        let a = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 5.0, 1.0);
        let b = SurfaceGeom::torus(PVec3::ZERO, PVec3::Z, 5.0, 1.0);
        assert!(same_surface(&a, &b, 1e-6));

        let c = SurfaceGeom::Torus {
            center: PVec3::ZERO,
            axis: PVec3::Z,
            major_r: 3.0,
            minor_r: 1.0,
            x_dir: PVec3::X,
            y_dir: PVec3::Y,
        };
        assert!(!same_surface(&a, &c, 1e-6));
    }
}
