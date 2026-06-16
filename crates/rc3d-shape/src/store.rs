//! B-Rep shape store with SlotMap storage. T1.5-T1.6

use std::collections::HashMap;
use slotmap::SlotMap;
use rc3d_core::math::{Real, PVec3};
use rc3d_core::utils::spatial::SpatialIndexF64;
use crate::topo::*;
use crate::tolerance::ToleranceContext;
use crate::geom::CurveGeom;
use crate::geom::SurfaceGeom;
use crate::geom::SurfaceParamRange;
use crate::geom::curve2d::Curve2d;
use crate::geom::normalize_edge_curve_to_vertices;

/// Canonical B-Rep topology storage (OCC `TopoDS_TShape` layer).
#[derive(Debug)]
pub struct BRepStore {
    pub vertices: SlotMap<VertexKey, BRepVertex>,
    pub edges: SlotMap<EdgeKey, BRepEdge>,
    pub wires: SlotMap<WireKey, BRepWire>,
    pub faces: SlotMap<FaceKey, BRepFace>,
    pub shells: SlotMap<ShellKey, BRepShell>,
    pub solids: SlotMap<SolidKey, BRepSolid>,
    pub compounds: SlotMap<CompoundKey, BRepCompound>,
    /// Spatial grid → VertexKey for tolerance-aware vertex deduplication.
    pub vertex_spatial_index: SpatialIndexF64<VertexKey>,
    /// Ordered endpoint pair → EdgeKey for O(1) edge deduplication.
    pub edge_hash_index: HashMap<(VertexKey, VertexKey), Vec<EdgeKey>>,
    /// Inverted index: edge → faces that reference this edge.
    /// Auto-maintained by add_edge_with_pcurve / add_seam_edge.
    /// Values may contain duplicates if a face references the same edge
    /// multiple times; call dedup_edge_to_faces() after bulk construction if needed.
    pub edge_to_faces: HashMap<EdgeKey, Vec<FaceKey>>,
    /// Inverted index: vertex → edges that reference this vertex.
    /// Auto-maintained (no explicit build call needed).
    pub vertex_to_edges: HashMap<VertexKey, Vec<EdgeKey>>,
    /// Model/heal/mesh distance tolerances for this shape document.
    pub tolerance: ToleranceContext,
    /// Per-face UV trim ranges from RECTANGULAR_TRIMMED_SURFACE / CURVE_BOUNDED_SURFACE.
    /// Keyed by FaceKey; (u_min, u_max, v_min, v_max). When absent, full domain is used.
    pub trim_ranges: HashMap<FaceKey, (Real, Real, Real, Real)>,
}

impl Default for BRepStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BRepStore {
    pub fn new() -> Self {
        Self::with_tolerance(ToleranceContext::default())
    }

    pub fn with_tolerance(tolerance: ToleranceContext) -> Self {
        let cell = tolerance.vertex_cell_size();
        Self {
            vertices: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            wires: SlotMap::with_key(),
            faces: SlotMap::with_key(),
            shells: SlotMap::with_key(),
            solids: SlotMap::with_key(),
            compounds: SlotMap::with_key(),
            vertex_spatial_index: SpatialIndexF64::with_cell_size(cell),
            edge_hash_index: HashMap::new(),
            edge_to_faces: HashMap::new(),
            vertex_to_edges: HashMap::new(),
            tolerance,
            trim_ranges: HashMap::new(),
        }
    }

    /// Insert or find an existing vertex at the given position within tolerance.
    pub fn find_or_add_vertex(&mut self, position: PVec3, tolerance: Real) -> VertexKey {
        let p = [position.x, position.y, position.z];
        if let Some(key) = self.vertex_spatial_index.find_near(p, tolerance, |k| {
            self.vertices
                .get(k)
                .map(|v| {
                    let pos = v.position;
                    [pos.x, pos.y, pos.z]
                })
                .unwrap_or([0.0; 3])
        }) {
            return key;
        }
        let key = self.vertices.insert(BRepVertex { position, tolerance });
        self.vertex_spatial_index.insert(key, p);
        key
    }

    /// Insert an edge with its PCURVE for a given face, or find an existing edge
    /// sharing the same endpoints and add the PCURVE to it.
    ///
    /// `v_start` / `v_end` define the ordered edge direction. Edge deduplication
    /// matches on the ordered `(v_start, v_end)` pair.
    ///
    /// `same_sense`: whether the supplied PCurve parameterization matches the
    /// 3D curve direction. When `false`, the PCurve is reversed before storage
    /// so every stored PCurve runs forward along the 3D curve.
    pub fn add_edge_with_pcurve(
        &mut self,
        v_start: VertexKey,
        v_end: VertexKey,
        curve: CurveGeom,
        tolerance: Real,
        face: FaceKey,
        pcurve: Curve2d,
        same_sense: bool,
    ) -> EdgeKey {
        let (v_lo, v_hi) = if v_start < v_end { (v_start, v_end) } else { (v_end, v_start) };
        let p_lo = self.vertices.get(v_lo).map(|v| v.position).unwrap_or(PVec3::ZERO);
        let p_hi = self.vertices.get(v_hi).map(|v| v.position).unwrap_or(PVec3::ZERO);
        let curve = normalize_edge_curve_to_vertices(curve, p_lo, p_hi, tolerance);
        // Normalize: store PCurve always in the forward (3D curve) direction.
        let pcurve = if same_sense { pcurve } else { pcurve.reversed() };

        // Check existing edges between same vertices — only reuse if curves match.
        // Sample at 7 Chebyshev-distributed points to reduce false-positive
        // merges for curves that share endpoints but are geometrically different.
        let chord_len = (p_hi - p_lo).length().max(tolerance);
        if let Some(existing) = self.edge_hash_index.get(&(v_lo, v_hi)) {
            'edge_check: for &ek in existing {
                for &t in &[0.038, 0.146, 0.309, 0.5, 0.691, 0.854, 0.962] {
                    let p_new = curve.d0(t);
                    let p_existing = self.edges.get(ek).map(|e| e.curve.d0(t)).unwrap_or(PVec3::ZERO);
                    let dist = (p_new - p_existing).length();
                    if dist > chord_len * 0.005 + tolerance * 2.0 {
                        continue 'edge_check;
                    }
                }
                // All sample points match — reuse existing edge
                if let Some(edge) = self.edges.get_mut(ek) {
                    edge.pcurves.insert(face, pcurve.clone());
                }
                // Auto-maintain edge_to_faces index (dedup)
                let v = self.edge_to_faces.entry(ek).or_default();
                if !v.contains(&face) { v.push(face); }
                return ek;
            }
        }
        // No matching curve — create a new edge
        let ek = self.edges.insert(BRepEdge {
            curve,
            tolerance,
            v_low: v_lo,
            v_high: v_hi,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: {
                let mut m = HashMap::new();
                m.insert(face, pcurve.clone());
                m
            },
        });
        self.edge_hash_index.entry((v_lo, v_hi)).or_default().push(ek);
        // Auto-maintain edge_to_faces and vertex_to_edges indices (dedup)
        { let v = self.edge_to_faces.entry(ek).or_default(); if !v.contains(&face) { v.push(face); } }
        self.vertex_to_edges.entry(v_lo).or_default().push(ek);
        self.vertex_to_edges.entry(v_hi).or_default().push(ek);
        ek
    }

    /// Insert a seam edge without endpoint deduplication (supports closed seams v_start == v_end).
    pub fn add_seam_edge(
        &mut self,
        v_start: VertexKey,
        v_end: VertexKey,
        curve: CurveGeom,
        tolerance: Real,
        face: FaceKey,
        pcurve: Curve2d,
        same_sense: bool,
    ) -> EdgeKey {
        let (v_lo, v_hi) = if v_start <= v_end {
            (v_start, v_end)
        } else {
            (v_end, v_start)
        };
        let p_lo = self.vertices.get(v_lo).map(|v| v.position).unwrap_or(PVec3::ZERO);
        let p_hi = self.vertices.get(v_hi).map(|v| v.position).unwrap_or(PVec3::ZERO);
        let curve = if v_lo == v_hi {
            curve
        } else {
            normalize_edge_curve_to_vertices(curve, p_lo, p_hi, tolerance)
        };
        let pcurve = if same_sense { pcurve } else { pcurve.reversed() };
        let ek = self.edges.insert(BRepEdge {
            curve,
            tolerance,
            v_low: v_lo,
            v_high: v_hi,
            t_min: 0.0,
            t_max: 1.0,
            pcurves: {
                let mut m = HashMap::new();
                m.insert(face, pcurve);
                m
            },
        });
        // Auto-maintain edge_to_faces and vertex_to_edges indices (dedup)
        { let v = self.edge_to_faces.entry(ek).or_default(); if !v.contains(&face) { v.push(face); } }
        self.vertex_to_edges.entry(v_lo).or_default().push(ek);
        if v_lo != v_hi {
            self.vertex_to_edges.entry(v_hi).or_default().push(ek);
        }
        ek
    }

    /// Insert a face with the given surface and an empty outer wire.
    /// Convenience builder to avoid repeating the BRepFace boilerplate.
    pub fn add_face(
        &mut self,
        surface: crate::geom::SurfaceGeom,
        tolerance: Real,
    ) -> FaceKey {
        self.faces.insert(BRepFace {
            surface,
            outer_wire: self.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![],
            same_sense: true,
            tolerance,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        })
    }

    /// Find edges shared by two faces. A shared edge has PCURVEs for both faces.
    /// O(edges) iteration using pcurve presence as the authoritative check.
    pub fn find_shared_edges(&self, face_a: FaceKey, face_b: FaceKey) -> Vec<EdgeKey> {
        // Fast path: iterate face_a's wire edges (works when wires are populated)
        let mut shared = Vec::new();
        if let Some(face) = self.faces.get(face_a) {
            let has_wire_edges = self.wires.get(face.outer_wire)
                .map(|w| !w.edges.is_empty())
                .unwrap_or(false);
            if has_wire_edges {
                for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
                    let Some(wire) = self.wires.get(*wire_key) else { continue };
                    for &(ek, _) in &wire.edges {
                        if let Some(edge) = self.edges.get(ek) {
                            if edge.pcurves.contains_key(&face_b) {
                                shared.push(ek);
                            }
                        }
                    }
                }
                return shared;
            }
        }
        // Fallback: iterate all edges (for test/legacy cases where wires are empty)
        for (ek, edge) in self.edges.iter() {
            if edge.pcurves.contains_key(&face_a) && edge.pcurves.contains_key(&face_b) {
                shared.push(ek);
            }
        }
        shared
    }

    /// Get mutable access to an edge's PCurve for a specific face.
    pub fn pcurve_mut(&mut self, ek: EdgeKey, face_key: FaceKey) -> Option<&mut Curve2d> {
        self.edges.get_mut(ek).and_then(|e| e.pcurves.get_mut(&face_key))
    }

    /// Replace or insert a PCurve for an (edge, face) pair.
    /// The PCurve is always stored normalized (forward = same direction as 3D curve).
    /// Returns the old PCurve if one existed.
    pub fn set_pcurve(&mut self, ek: EdgeKey, face_key: FaceKey, pcurve: Curve2d, same_sense: bool) -> Option<Curve2d> {
        let pcurve = if same_sense { pcurve } else { pcurve.reversed() };
        self.edges.get_mut(ek).and_then(|e| e.pcurves.insert(face_key, pcurve))
    }

    pub fn iter_faces(&self) -> impl Iterator<Item = (FaceKey, &BRepFace)> {
        self.faces.iter()
    }

    /// Deduplicate face entries in edge_to_faces. Only needed after bulk
    /// construction that bypassed add_edge_with_pcurve (e.g., direct SlotMap
    /// insertion during repair operations).
    pub fn dedup_edge_to_faces(&mut self) {
        for faces in self.edge_to_faces.values_mut() {
            faces.sort();
            faces.dedup();
        }
    }

    /// Rebuild edge_to_faces from scratch by scanning all face wires.
    /// Prefer the auto-maintained index (add_edge_with_pcurve / add_seam_edge
    /// now populate it automatically). This method is retained for bulk repair
    /// scenarios where edges were inserted directly into the SlotMap.
    pub fn build_edge_to_faces_index(&mut self) {
        let mut index: HashMap<EdgeKey, Vec<FaceKey>> = HashMap::new();
        for (fk, face) in self.faces.iter() {
            for wire_key in std::iter::once(&face.outer_wire).chain(face.inner_wires.iter()) {
                let Some(wire) = self.wires.get(*wire_key) else { continue };
                for &(ek, _) in &wire.edges {
                    index.entry(ek).or_default().push(fk);
                }
            }
        }
        // Deduplicate
        for faces in index.values_mut() {
            faces.sort();
            faces.dedup();
        }
        self.edge_to_faces = index;
    }
}

impl BRepStore {
    /// Denormalize [0,1]^2 UV coordinates to the face's trim range (native STEP space).
    /// When no trim range is stored for this face, returns (u, v) unchanged.
    pub fn face_native_uv(&self, face_key: FaceKey, u: Real, v: Real) -> (Real, Real) {
        if let Some(&(u_min, u_max, v_min, v_max)) = self.trim_ranges.get(&face_key) {
            let du = (u_max - u_min).max(1e-12);
            let dv = (v_max - v_min).max(1e-12);
            (u_min + u * du, v_min + v * dv)
        } else {
            (u, v)
        }
    }

    /// Effective parameter range for a face: trim range if stored, otherwise surface.param_range().
    pub fn face_param_range(&self, face_key: FaceKey, surface: &SurfaceGeom) -> SurfaceParamRange {
        if let Some(&(u_min, u_max, v_min, v_max)) = self.trim_ranges.get(&face_key) {
            SurfaceParamRange { u_min, u_max, v_min, v_max }
        } else {
            surface.param_range()
        }
    }
}

/// Lightweight wrapper for PCurve modification during healing (OCC equivalent).
/// Enables batched PCurve edits with audit trail.
#[derive(Debug, Clone)]
pub struct PCurveEdit {
    pub edge_key: crate::topo::EdgeKey,
    pub face_key: crate::topo::FaceKey,
    /// Replacement PCurve. None = remove existing.
    pub new_pcurve: Option<Curve2d>,
}

impl BRepStore {
    /// Apply a PCurveEdit to the store. Returns the old PCurve if replaced.
    pub fn apply_pcurve_edit(&mut self, edit: &PCurveEdit) -> Option<Curve2d> {
        let ek = edit.edge_key;
        let fk = edit.face_key;
        match &edit.new_pcurve {
            Some(pc) => self.set_pcurve(ek, fk, pc.clone(), true),
            None => self.edges.get_mut(ek).and_then(|e| e.pcurves.remove(&fk)),
        }
    }

    /// OCC BRepLib::SameParameter — ensure that for all t ∈ [0,1]:
    ///   | surface(pcurve(t)) - curve3d(t) | < tolerance
    ///
    /// Iteratively adjusts the PCurve control points by projecting 3D
    /// curve samples onto the face surface. After convergence, edge
    /// tolerance is raised if residual deviation exceeds the current
    /// edge tolerance.
    pub fn ensure_same_parameter(
        &mut self,
        ek: EdgeKey,
        fk: FaceKey,
        tolerance: Real,
        max_iterations: usize,
    ) -> Option<SameParamResult> {
        let curve_3d = self.edges.get(ek)?.curve.clone();
        // Circle edges: the 3D curve already matches the surface geometry.
        // Skip PCurve adjustment to avoid phase-mismatch false deviation.
        if matches!(curve_3d, CurveGeom::Circle { .. }) {
            return Some(SameParamResult {
                deviation_before: 0.0, deviation_after: 0.0,
                converged: true, iterations: 0, tolerance_updated: false,
            });
        }
        let pcurve = self.edges.get(ek)?.pcurves.get(&fk)?.clone();
        let surface = self.faces.get(fk)?.surface.clone();
        let edge_tolerance = self.edges.get(ek).map(|e| e.tolerance).unwrap_or(tolerance);

        const SAMPLE_COUNT: usize = 32;

        let before = max_pcurve_deviation(&curve_3d, &pcurve, &surface, SAMPLE_COUNT);
        if before <= tolerance {
            return Some(SameParamResult {
                deviation_before: before,
                deviation_after: before,
                converged: true,
                iterations: 0,
                tolerance_updated: false,
            });
        }

        let mut current = pcurve;
        let mut converged = false;
        let mut iters = 0usize;

        for i in 0..max_iterations {
            iters = i + 1;
            let samples = sample_pcurve_deviations(&curve_3d, &current, &surface, SAMPLE_COUNT);
            if let Some(adjusted) = adjust_pcurve_from_samples(&current, &samples, &surface) {
                current = adjusted;
            } else {
                break;
            }
            let after = max_pcurve_deviation(&curve_3d, &current, &surface, SAMPLE_COUNT);
            if after <= tolerance {
                converged = true;
                break;
            }
        }

        let after = max_pcurve_deviation(&curve_3d, &current, &surface, SAMPLE_COUNT);

        // Store the adjusted PCurve
        if let Some(edge_mut) = self.edges.get_mut(ek) {
            edge_mut.pcurves.insert(fk, current);
        }

        // Raise edge tolerance if residual deviation exceeds current edge tolerance.
        // Clamp to the chord length to prevent runaway inflation when the 3D curve
        // and PCurve represent fundamentally different geometries (e.g. SURFACE_CURVE
        // with a Line 3D curve hiding a circle).
        let chord_len = {
            let lo = self.vertices.get(self.edges.get(ek)?.v_low).map(|v| v.position);
            let hi = self.vertices.get(self.edges.get(ek)?.v_high).map(|v| v.position);
            match (lo, hi) {
                (Some(l), Some(h)) => (h - l).length().max(1e-6),
                _ => 1e-6,
            }
        };
        let new_tol = after.max(edge_tolerance).max(tolerance).min(chord_len);
        let tolerance_updated = if new_tol > edge_tolerance + 1e-8 {
            if let Some(edge_mut) = self.edges.get_mut(ek) {
                edge_mut.tolerance = new_tol;
            }
            true
        } else {
            false
        };

        Some(SameParamResult {
            deviation_before: before,
            deviation_after: after,
            converged,
            iterations: iters,
            tolerance_updated,
        })
    }
}

/// Result of ensuring PCurve-on-surface alignment with 3D curve.
#[derive(Debug, Clone)]
pub struct SameParamResult {
    pub deviation_before: Real,
    pub deviation_after: Real,
    pub converged: bool,
    pub iterations: usize,
    pub tolerance_updated: bool,
}

// ── SameParameter helpers (shared) ─────────────────────────────────

struct PcDevSample {
    #[allow(dead_code)] t: Real,
    pt_3d: rc3d_core::math::PVec3,
    dev: Real,
}

fn sample_pcurve_deviations(
    curve_3d: &CurveGeom, pcurve: &Curve2d, surface: &SurfaceGeom, n: usize,
) -> Vec<PcDevSample> {
    use crate::geom::SurfaceGeom;
    (0..=n).map(|i| {
        let t = i as Real / n as Real;
        let pt_3d = curve_3d.d0(t);
        let uv = pcurve.d0(t);
        let on_surf = surface.d0_native(uv.0, uv.1);
        PcDevSample { t, pt_3d, dev: (on_surf - pt_3d).length() }
    }).collect()
}

fn max_pcurve_deviation(
    curve_3d: &CurveGeom, pcurve: &Curve2d, surface: &SurfaceGeom, n: usize,
) -> Real {
    sample_pcurve_deviations(curve_3d, pcurve, surface, n)
        .iter().map(|s| s.dev).fold(0.0_f64, Real::max)
}

fn adjust_pcurve_from_samples(
    pcurve: &Curve2d, samples: &[PcDevSample], surface: &SurfaceGeom,
) -> Option<Curve2d> {
    use crate::geom::SurfaceGeom;
    match pcurve {
        Curve2d::Line { .. } => {
            let first = &samples[0];
            let last = &samples[samples.len() - 1];
            let uv_start = surface.project(first.pt_3d)?;
            let uv_end = surface.project(last.pt_3d)?;
            let d = (uv_end.0 - uv_start.0, uv_end.1 - uv_start.1);
            if d.0.abs() < 1e-12 && d.1.abs() < 1e-12 { return None; }
            Some(Curve2d::Line { origin: uv_start, direction: d })
        }
        Curve2d::Polyline { points } => {
            if points.len() < 2 || samples.is_empty() { return None; }
            let step = (samples.len().saturating_sub(1) / points.len().saturating_sub(1)).max(1);
            let new_pts: Vec<(Real, Real)> = points.iter().enumerate().map(|(idx, &orig)| {
                let si = (idx * step).min(samples.len() - 1);
                surface.project(samples[si].pt_3d).unwrap_or(orig)
            }).collect();
            Some(Curve2d::Polyline { points: new_pts })
        }
        Curve2d::BSpline { degree, control_points, knots, weights } => {
            let n = control_points.len();
            if n < 2 { return None; }
            let first_3d = &samples[0].pt_3d;
            let last_3d = &samples[samples.len() - 1].pt_3d;
            let uv_first = surface.project(*first_3d)?;
            let uv_last = surface.project(*last_3d)?;
            let mut new_cps = control_points.clone();
            new_cps[0] = uv_first;
            new_cps[n - 1] = uv_last;
            let step = (samples.len().saturating_sub(1) / n.saturating_sub(1)).max(1);
            for i in 1..n - 1 {
                let si = (i * step).min(samples.len() - 1);
                if let Some(uv) = surface.project(samples[si].pt_3d) {
                    new_cps[i] = ((new_cps[i].0 + uv.0) * 0.5, (new_cps[i].1 + uv.1) * 0.5);
                }
            }
            Some(Curve2d::BSpline {
                degree: *degree, control_points: new_cps,
                knots: knots.clone(), weights: weights.clone(),
            })
        }
        // Circle/Ellipse/Trimmed/Composite → polyline approximation
        _ => {
            let n_samples = 16usize;
            let pts: Vec<(Real, Real)> = (0..=n_samples).filter_map(|i| {
                let t = i as Real / n_samples as Real;
                let si = (t * (samples.len() - 1) as Real) as usize;
                surface.project(samples[si.min(samples.len() - 1)].pt_3d)
            }).collect();
            if pts.len() < 2 { return None; }
            Some(Curve2d::Polyline { points: pts })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::PVec3;
    use crate::geom::SurfaceGeom;

    fn make_plane_face(reg: &mut BRepStore) -> FaceKey {
        reg.add_face(
            SurfaceGeom::Plane { origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X },
            1e-4,
        )
    }

    #[test]
    fn test_find_or_add_vertex_dedup() {
        let mut reg = BRepStore::new();
        let a = reg.find_or_add_vertex(PVec3::new(1.0, 2.0, 3.0), 1e-4);
        let b = reg.find_or_add_vertex(PVec3::new(1.0, 2.0, 3.0), 1e-4);
        assert_eq!(a, b);
        assert_eq!(reg.vertices.len(), 1);
    }

    #[test]
    fn test_find_or_add_vertex_near_duplicate_within_tolerance() {
        let mut reg = BRepStore::new();
        let a = reg.find_or_add_vertex(PVec3::new(1.0, 2.0, 3.0), 1e-3);
        let b = reg.find_or_add_vertex(PVec3::new(1.0 + 5e-4, 2.0, 3.0), 1e-3);
        assert_eq!(a, b);
        assert_eq!(reg.vertices.len(), 1);
    }

    #[test]
    fn test_find_or_add_vertex_distinct_beyond_tolerance() {
        let mut reg = BRepStore::new();
        let a = reg.find_or_add_vertex(PVec3::ZERO, 1e-5);
        let b = reg.find_or_add_vertex(PVec3::new(1e-4, 0.0, 0.0), 1e-5);
        assert_ne!(a, b);
        assert_eq!(reg.vertices.len(), 2);
    }

    #[test]
    fn test_add_edge_with_pcurve_dedup() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);

        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let e0 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);

        assert_eq!(e0, e1, "same endpoint pair should return same edge");
        let edge = reg.edges.get(e0).unwrap();
        assert!(edge.pcurves.contains_key(&f0));
        assert!(edge.pcurves.contains_key(&f1));
    }

    #[test]
    fn test_find_shared_edges() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);
        let f2 = make_plane_face(&mut reg);

        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);

        let shared = reg.find_shared_edges(f0, f1);
        assert_eq!(shared.len(), 1, "f0 and f1 share one edge");
        let not_shared = reg.find_shared_edges(f0, f2);
        assert!(not_shared.is_empty(), "f0 and f2 share no edges");
    }

    #[test]
    fn test_set_pcurve_replace() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);

        let new_pcurve = Curve2d::Line { origin: (1.0, 0.0), direction: (1.0, 0.0) };
        let old = reg.set_pcurve(ek, f0, new_pcurve, true);
        assert!(old.is_some());

        let edge = reg.edges.get(ek).unwrap();
        let pcurve = edge.pcurves.get(&f0).unwrap();
        match pcurve {
            Curve2d::Line { origin, .. } => {
                assert!((origin.0 - 1.0).abs() < 1e-6, "expected new pcurve origin");
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn test_pcurve_mut() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);

        let pcurve = reg.pcurve_mut(ek, f0).unwrap();
        *pcurve = Curve2d::Line { origin: (2.0, 0.0), direction: (0.0, 1.0) };
        // Explicitly end the mutable borrow before reading back
        let _ = pcurve;

        let edge = reg.edges.get(ek).unwrap();
        let updated = edge.pcurves.get(&f0).unwrap();
        match updated {
            Curve2d::Line { origin, direction } => {
                assert!((origin.0 - 2.0).abs() < 1e-6);
                assert!((direction.1 - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn test_set_pcurve_new_face() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);

        let old = reg.set_pcurve(ek, f1, pc.clone(), true);
        assert!(old.is_none(), "f1 had no pcurve before");

        let edge = reg.edges.get(ek).unwrap();
        assert!(edge.pcurves.contains_key(&f0));
        assert!(edge.pcurves.contains_key(&f1));
    }

    #[test]
    fn test_vertex_to_edges_index() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::Y, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let line2 = CurveGeom::Line { origin: PVec3::X, direction: PVec3::Y - PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let pc2 = Curve2d::Line { origin: (1.0, 0.0), direction: (-1.0, 1.0) };

        let e0 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc, true);
        let e1 = reg.add_edge_with_pcurve(v1, v2, line2.clone(), 1e-4, f0, pc2, true);

        let v0_edges = reg.vertex_to_edges.get(&v0).unwrap();
        assert_eq!(v0_edges, &vec![e0], "v0 belongs only to e0");
        let v1_edges = reg.vertex_to_edges.get(&v1).unwrap();
        assert_eq!(v1_edges.len(), 2, "v1 belongs to both e0 and e1");
        assert!(v1_edges.contains(&e0));
        assert!(v1_edges.contains(&e1));
    }

    #[test]
    fn test_dual_index_consistency() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };

        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, pc.clone(), true);
        // Adding same edge from f1 should reuse ek
        let ek2 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, pc.clone(), true);
        assert_eq!(ek, ek2, "should reuse existing edge");

        // edge_to_faces should have both faces
        let faces = reg.edge_to_faces.get(&ek).unwrap();
        assert!(faces.contains(&f0), "edge_to_faces should contain f0");
        assert!(faces.contains(&f1), "edge_to_faces should contain f1");

        // vertex_to_edges should reference ek
        assert!(reg.vertex_to_edges.get(&v0).unwrap().contains(&ek));
        assert!(reg.vertex_to_edges.get(&v1).unwrap().contains(&ek));
    }

    #[test]
    fn test_seam_edge_indices() {
        let mut reg = BRepStore::new();
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };

        let ek = reg.add_seam_edge(v0, v0, line.clone(), 1e-4, f0, pc, true);

        // vertex_to_edges should have v0 → ek (only once, not duplicated)
        let v0_edges = reg.vertex_to_edges.get(&v0).unwrap();
        assert!(v0_edges.contains(&ek), "seam edge should be in vertex_to_edges");
        // edge_to_faces should have the face
        assert!(reg.edge_to_faces.get(&ek).unwrap().contains(&f0));
    }
}
