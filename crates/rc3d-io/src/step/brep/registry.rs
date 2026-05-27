//! B-Rep shape registry with SlotMap storage. T1.5-T1.6

use std::collections::HashMap;
use slotmap::SlotMap;
use rc3d_core::math::Vec3;
use rc3d_core::utils::hash::f32x3_quantized_bits;
use super::topo::*;
use super::geom::CurveGeom;
use super::geom::normalize_edge_curve_to_vertices;

#[derive(Debug)]
pub struct BRepRegistry {
    pub vertices: SlotMap<VertexKey, BRepVertex>,
    pub edges: SlotMap<EdgeKey, BRepEdge>,
    pub wires: SlotMap<WireKey, BRepWire>,
    pub faces: SlotMap<FaceKey, BRepFace>,
    pub shells: SlotMap<ShellKey, BRepShell>,
    pub solids: SlotMap<SolidKey, BRepSolid>,
    /// Spatial hash → VertexKey for O(1) vertex deduplication.
    pub vertex_hash_index: HashMap<[u32; 3], VertexKey>,
    /// Ordered endpoint pair → EdgeKey for O(1) edge deduplication.
    pub edge_hash_index: HashMap<(VertexKey, VertexKey), EdgeKey>,
}

impl BRepRegistry {
    pub fn new() -> Self {
        Self {
            vertices: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            wires: SlotMap::with_key(),
            faces: SlotMap::with_key(),
            shells: SlotMap::with_key(),
            solids: SlotMap::with_key(),
            vertex_hash_index: HashMap::new(),
            edge_hash_index: HashMap::new(),
        }
    }

    /// Insert or find an existing vertex at the given position within tolerance.
    /// Uses `f32x3_quantized_bits` spatial hashing for O(1) lookup.
    pub fn find_or_add_vertex(&mut self, position: Vec3, tolerance: f32) -> VertexKey {
        let hash = f32x3_quantized_bits([position.x, position.y, position.z]);
        if let Some(&key) = self.vertex_hash_index.get(&hash) {
            if let Some(v) = self.vertices.get(key) {
                if (v.position - position).length() <= tolerance {
                    return key;
                }
            }
        }
        let key = self.vertices.insert(BRepVertex { position, tolerance });
        self.vertex_hash_index.insert(hash, key);
        key
    }

    /// Insert an edge with its PCURVE for a given face, or find an existing edge
    /// sharing the same endpoints and add the PCURVE to it.
    ///
    /// `v_start` / `v_end` define the ordered edge direction. Edge deduplication
    /// matches on the ordered `(v_start, v_end)` pair.
    pub fn add_edge_with_pcurve(
        &mut self,
        v_start: VertexKey,
        v_end: VertexKey,
        curve: CurveGeom,
        tolerance: f32,
        face: FaceKey,
        pcurve: CurveGeom,
    ) -> EdgeKey {
        let (v_lo, v_hi) = if v_start < v_end { (v_start, v_end) } else { (v_end, v_start) };
        if let Some(&ek) = self.edge_hash_index.get(&(v_lo, v_hi)) {
            if let Some(edge) = self.edges.get_mut(ek) {
                edge.pcurves.insert(face, pcurve);
            }
            return ek;
        }
        let p_lo = self.vertices.get(v_lo).map(|v| v.position).unwrap_or(Vec3::ZERO);
        let p_hi = self.vertices.get(v_hi).map(|v| v.position).unwrap_or(Vec3::ZERO);
        let curve = normalize_edge_curve_to_vertices(curve, p_lo, p_hi, tolerance);
        let ek = self.edges.insert(BRepEdge {
            curve,
            tolerance,
            v_low: v_lo,
            v_high: v_hi,
            pcurves: {
                let mut m = HashMap::new();
                m.insert(face, pcurve);
                m
            },
        });
        self.edge_hash_index.insert((v_lo, v_hi), ek);
        ek
    }

    /// Insert a seam edge without endpoint deduplication (supports closed seams v_start == v_end).
    pub fn add_seam_edge(
        &mut self,
        v_start: VertexKey,
        v_end: VertexKey,
        curve: CurveGeom,
        tolerance: f32,
        face: FaceKey,
        pcurve: CurveGeom,
    ) -> EdgeKey {
        let (v_lo, v_hi) = if v_start <= v_end {
            (v_start, v_end)
        } else {
            (v_end, v_start)
        };
        let p_lo = self.vertices.get(v_lo).map(|v| v.position).unwrap_or(Vec3::ZERO);
        let p_hi = self.vertices.get(v_hi).map(|v| v.position).unwrap_or(Vec3::ZERO);
        let curve = if v_lo == v_hi {
            curve
        } else {
            normalize_edge_curve_to_vertices(curve, p_lo, p_hi, tolerance)
        };
        self.edges.insert(BRepEdge {
            curve,
            tolerance,
            v_low: v_lo,
            v_high: v_hi,
            pcurves: {
                let mut m = HashMap::new();
                m.insert(face, pcurve);
                m
            },
        })
    }

    /// Find edges shared by two faces. A shared edge has PCURVEs for both faces.
    pub fn find_shared_edges(&self, face_a: FaceKey, face_b: FaceKey) -> Vec<EdgeKey> {
        let mut shared = Vec::new();
        for (ek, edge) in self.edges.iter() {
            let has_a = edge.pcurves.contains_key(&face_a);
            let has_b = edge.pcurves.contains_key(&face_b);
            if has_a && has_b {
                shared.push(ek);
            }
        }
        shared
    }

    /// Get mutable access to an edge's PCurve for a specific face.
    pub fn pcurve_mut(&mut self, ek: EdgeKey, face_key: FaceKey) -> Option<&mut CurveGeom> {
        self.edges.get_mut(ek).and_then(|e| e.pcurves.get_mut(&face_key))
    }

    /// Replace or insert a PCurve for an (edge, face) pair.
    /// Returns the old PCurve if one existed.
    pub fn set_pcurve(&mut self, ek: EdgeKey, face_key: FaceKey, pcurve: CurveGeom) -> Option<CurveGeom> {
        self.edges.get_mut(ek).and_then(|e| e.pcurves.insert(face_key, pcurve))
    }

    pub fn iter_faces(&self) -> impl Iterator<Item = (FaceKey, &BRepFace)> {
        self.faces.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rc3d_core::math::Vec3;
    use super::super::geom::SurfaceGeom;

    fn make_plane_face(reg: &mut BRepRegistry) -> FaceKey {
        reg.faces.insert(BRepFace {
            surface: SurfaceGeom::Plane { origin: Vec3::ZERO, normal: Vec3::Z, u_dir: Vec3::X },
            outer_wire: reg.wires.insert(BRepWire { edges: vec![] }),
            inner_wires: vec![], same_sense: true, tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        })
    }

    #[test]
    fn test_find_or_add_vertex_dedup() {
        let mut reg = BRepRegistry::new();
        let a = reg.find_or_add_vertex(Vec3::new(1.0, 2.0, 3.0), 1e-4);
        let b = reg.find_or_add_vertex(Vec3::new(1.0, 2.0, 3.0), 1e-4);
        assert_eq!(a, b);
        assert_eq!(reg.vertices.len(), 1);
    }

    #[test]
    fn test_add_edge_with_pcurve_dedup() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let e0 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());
        let e1 = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, line.clone());

        assert_eq!(e0, e1, "same endpoint pair should return same edge");
        let edge = reg.edges.get(e0).unwrap();
        assert!(edge.pcurves.contains_key(&f0));
        assert!(edge.pcurves.contains_key(&f1));
    }

    #[test]
    fn test_find_shared_edges() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);
        let f2 = make_plane_face(&mut reg);

        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());
        reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f1, line.clone());

        let shared = reg.find_shared_edges(f0, f1);
        assert_eq!(shared.len(), 1, "f0 and f1 share one edge");
        let not_shared = reg.find_shared_edges(f0, f2);
        assert!(not_shared.is_empty(), "f0 and f2 share no edges");
    }

    #[test]
    fn test_set_pcurve_replace() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

        let new_pcurve = CurveGeom::Line { origin: Vec3::new(1.0, 0.0, 0.0), direction: Vec3::X };
        let old = reg.set_pcurve(ek, f0, new_pcurve.clone());
        assert!(old.is_some());

        let edge = reg.edges.get(ek).unwrap();
        let pcurve = edge.pcurves.get(&f0).unwrap();
        match pcurve {
            CurveGeom::Line { origin, .. } => {
                assert!((origin.x - 1.0).abs() < 1e-6, "expected new pcurve origin");
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn test_pcurve_mut() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

        let pc = reg.pcurve_mut(ek, f0).unwrap();
        *pc = CurveGeom::Line { origin: Vec3::new(2.0, 0.0, 0.0), direction: Vec3::Y };
        drop(pc);

        let edge = reg.edges.get(ek).unwrap();
        let updated = edge.pcurves.get(&f0).unwrap();
        match updated {
            CurveGeom::Line { origin, direction } => {
                assert!((origin.x - 2.0).abs() < 1e-6);
                assert!((direction.y - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Line"),
        }
    }

    #[test]
    fn test_set_pcurve_new_face() {
        let mut reg = BRepRegistry::new();
        let v0 = reg.find_or_add_vertex(Vec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(Vec3::X, 1e-4);
        let f0 = make_plane_face(&mut reg);
        let f1 = make_plane_face(&mut reg);
        let line = CurveGeom::Line { origin: Vec3::ZERO, direction: Vec3::X };
        let ek = reg.add_edge_with_pcurve(v0, v1, line.clone(), 1e-4, f0, line.clone());

        let old = reg.set_pcurve(ek, f1, line.clone());
        assert!(old.is_none(), "f1 had no pcurve before");

        let edge = reg.edges.get(ek).unwrap();
        assert!(edge.pcurves.contains_key(&f0));
        assert!(edge.pcurves.contains_key(&f1));
    }
}
