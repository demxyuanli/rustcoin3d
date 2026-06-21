//! Open edge detection and closing (free bounds repair).
//!
//! Finds edges not shared by two faces and attempts to close them
//! by geometric matching of coincident open edges.
//!
//! OCC alignment: ShapeFix_FreeBounds — detects edges with < 2 face refs
//! and closes them via vertex merging or gap filling.

use crate::geom::curve2d::Curve2d;
use crate::store::BRepStore;
use crate::topo::*;
use rc3d_core::math::{Real, PVec3};

/// Result of free bounds analysis and repair.
#[derive(Debug, Clone, Default)]
pub struct FreeBoundsReport {
    pub open_edges_found: usize,
    pub edges_closed: usize,
}

/// Find all open edges in a shell — edges referenced by fewer than 2 faces.
///
/// Uses `reg.edge_to_faces` inverted index for O(E) scan.
pub fn find_open_edges(shell_key: ShellKey, reg: &BRepStore) -> Vec<EdgeKey> {
    let Some(shell) = reg.shells.get(shell_key) else {
        return Vec::new();
    };

    // Collect all edges referenced by this shell's faces
    let mut shell_edges = Vec::new();
    for &(fk, _) in &shell.faces {
        let Some(face) = reg.faces.get(fk) else { continue };
        for wire_key in std::iter::once(&face.outer_wire).chain(&face.inner_wires) {
            let Some(wire) = reg.wires.get(*wire_key) else { continue };
            for &(ek, _) in &wire.edges {
                shell_edges.push(ek);
            }
        }
    }

    // Open edges: referenced by < 2 faces
    shell_edges
        .iter()
        .filter(|&&ek| {
            let face_count = reg.edge_to_faces.get(&ek).map(|v| v.len()).unwrap_or(0);
            face_count < 2
        })
        .copied()
        .collect()
}

/// Attempt to close open edges by finding geometrically coincident partners.
///
/// Strategy:
/// 1. Find all open edges
/// 2. For each pair, check if endpoints are within tolerance
/// 3. Merge matching edges by adding the second face's PCurve to the surviving edge
pub fn close_free_bounds(
    shell_key: ShellKey,
    reg: &mut BRepStore,
    tolerance: Real,
) -> FreeBoundsReport {
    let open_edges = find_open_edges(shell_key, reg);
    let open_count = open_edges.len();
    if open_count == 0 {
        return FreeBoundsReport {
            open_edges_found: 0,
            ..Default::default()
        };
    }

    let mut report = FreeBoundsReport {
        open_edges_found: open_count,
        ..Default::default()
    };

    // Collect endpoint positions for matching
    let endpoints: Vec<(PVec3, PVec3)> = open_edges
        .iter()
        .filter_map(|&ek| {
            let edge = reg.edges.get(ek)?;
            let p_lo = reg.vertices.get(edge.v_low)?.position;
            let p_hi = reg.vertices.get(edge.v_high)?.position;
            Some((p_lo, p_hi))
        })
        .collect();

    // Try to match pairs of open edges with coincident endpoints
    let mut closed = vec![false; open_edges.len()];
    // Track merged edge pairs for wire update after the matching loop
    let mut merges: Vec<(EdgeKey, EdgeKey)> = Vec::new();

    for i in 0..open_edges.len() {
        if closed[i] {
            continue;
        }
        let (lo_i, hi_i) = match endpoints.get(i) {
            Some(e) => *e,
            None => continue,
        };

        for j in (i + 1)..open_edges.len() {
            if closed[j] {
                continue;
            }
            let (lo_j, hi_j) = match endpoints.get(j) {
                Some(e) => *e,
                None => continue,
            };

            // Check if edges have matching endpoints (forward or reversed)
            let match_forward = (lo_i - lo_j).length() < tolerance
                && (hi_i - hi_j).length() < tolerance;
            let match_reversed = (lo_i - hi_j).length() < tolerance
                && (hi_i - lo_j).length() < tolerance;

            if match_forward || match_reversed {
                // Gather faces referencing each edge
                let faces_i: Vec<FaceKey> = reg.edge_to_faces.get(&open_edges[i])
                    .cloned().unwrap_or_default();
                let faces_j: Vec<FaceKey> = reg.edge_to_faces.get(&open_edges[j])
                    .cloned().unwrap_or_default();

                // Select survivor (edge with more faces, or lower key for determinism)
                let (survivor, victim) = if faces_i.len() >= faces_j.len() {
                    (open_edges[i], open_edges[j])
                } else {
                    (open_edges[j], open_edges[i])
                };

                // Merge vertices: ensure both edges share the same endpoint vertices
                // so the topological edge closure is real, not just wire-level.
                {
                    let surv_p_lo = reg.edges.get(survivor)
                        .and_then(|e| reg.vertices.get(e.v_low))
                        .map(|v| v.position);
                    let surv_p_hi = reg.edges.get(survivor)
                        .and_then(|e| reg.vertices.get(e.v_high))
                        .map(|v| v.position);
                    let vict_p_lo = reg.edges.get(victim)
                        .and_then(|e| reg.vertices.get(e.v_low))
                        .map(|v| v.position);
                    let vict_p_hi = reg.edges.get(victim)
                        .and_then(|e| reg.vertices.get(e.v_high))
                        .map(|v| v.position);

                    if let (Some(sp_lo), Some(sp_hi), Some(vp_lo), Some(vp_hi)) =
                        (surv_p_lo, surv_p_hi, vict_p_lo, vict_p_hi)
                    {
                        // Merge endpoint vertices — find_or_add_vertex deduplicates
                        // within tolerance, returning a shared key for both edges.
                        let merged_v_lo = reg.find_or_add_vertex(sp_lo, tolerance);
                        let merged_v_hi = reg.find_or_add_vertex(sp_hi, tolerance);
                        // The victim's matching endpoint will also resolve to the same key
                        // because its position is within tolerance (checked above).
                        let _merged_vict_lo = reg.find_or_add_vertex(vp_lo, tolerance);
                        let _merged_vict_hi = reg.find_or_add_vertex(vp_hi, tolerance);

                        // Update survivor edge vertices
                        if let Some(se) = reg.edges.get_mut(survivor) {
                            let old_lo = se.v_low;
                            let old_hi = se.v_high;
                            se.v_low = merged_v_lo;
                            se.v_high = merged_v_hi;
                            // Maintain vertex_to_edges index
                            if old_lo != merged_v_lo {
                                if let Some(edges) = reg.vertex_to_edges.get_mut(&old_lo) {
                                    edges.retain(|&e| e != survivor);
                                }
                                reg.vertex_to_edges.entry(merged_v_lo).or_default().push(survivor);
                            }
                            if old_hi != merged_v_hi {
                                if let Some(edges) = reg.vertex_to_edges.get_mut(&old_hi) {
                                    edges.retain(|&e| e != survivor);
                                }
                                reg.vertex_to_edges.entry(merged_v_hi).or_default().push(survivor);
                            }
                        }

                        // Update victim edge vertices to match survivor
                        if let Some(ve) = reg.edges.get_mut(victim) {
                            let old_lo = ve.v_low;
                            let old_hi = ve.v_high;
                            let (new_lo, new_hi) = if match_forward {
                                (merged_v_lo, merged_v_hi)
                            } else {
                                (merged_v_hi, merged_v_lo)
                            };
                            ve.v_low = new_lo;
                            ve.v_high = new_hi;
                            // Update vertex_to_edges for victim
                            if old_lo != new_lo {
                                if let Some(edges) = reg.vertex_to_edges.get_mut(&old_lo) {
                                    edges.retain(|&e| e != victim);
                                }
                                reg.vertex_to_edges.entry(new_lo).or_default().push(victim);
                            }
                            if old_hi != new_hi {
                                if let Some(edges) = reg.vertex_to_edges.get_mut(&old_hi) {
                                    edges.retain(|&e| e != victim);
                                }
                                reg.vertex_to_edges.entry(new_hi).or_default().push(victim);
                            }
                        }
                    }
                }

                // Merge by sharing: copy PCurves from victim to survivor edge.

                // Copy PCurves from victim to survivor
                if let Some(victim_edge) = reg.edges.get(victim) {
                    let victim_pcurves: Vec<(FaceKey, Curve2d)> = victim_edge.pcurves
                        .iter()
                        .map(|(&fk, pc)| (fk, pc.clone()))
                        .collect();
                    if let Some(survivor_edge) = reg.edges.get_mut(survivor) {
                        for (fk, pc) in victim_pcurves {
                            survivor_edge.pcurves.entry(fk).or_insert(pc);
                        }
                    }
                }

                // Update edge_to_faces: merge victim's faces into survivor
                if let Some(victim_faces) = reg.edge_to_faces.get(&victim).cloned() {
                    let survivor_faces = reg.edge_to_faces.entry(survivor).or_default();
                    for fk in victim_faces {
                        if !survivor_faces.contains(&fk) {
                            survivor_faces.push(fk);
                        }
                    }
                }

                // Update vertex_to_edges: replace victim with survivor
                let vict_v_low = reg.edges.get(victim).map(|e| e.v_low);
                let vict_v_high = reg.edges.get(victim).map(|e| e.v_high);
                for vk in vict_v_low.iter().chain(vict_v_high.iter()) {
                    if let Some(edges) = reg.vertex_to_edges.get_mut(vk) {
                        if let Some(pos) = edges.iter().position(|&e| e == victim) {
                            edges[pos] = survivor;
                        }
                    }
                }

                // Record merge for wire update
                merges.push((victim, survivor));

                closed[i] = true;
                closed[j] = true;
                report.edges_closed += 1;
                break;
            }
        }
    }

    // Update face wires: replace victim edges with survivor edges
    let Some(shell) = reg.shells.get(shell_key) else { return report; };
    let face_keys: Vec<FaceKey> = shell.faces.iter().map(|&(fk, _)| fk).collect();
    for fk in face_keys {
        let Some(face) = reg.faces.get(fk) else { continue };
        let wire_keys: Vec<WireKey> = std::iter::once(face.outer_wire)
            .chain(face.inner_wires.iter().copied())
            .collect();
        for wk in wire_keys {
            if let Some(wire) = reg.wires.get_mut(wk) {
                for (ek_ref, _) in wire.edges.iter_mut() {
                    for &(victim, survivor) in &merges {
                        if *ek_ref == victim {
                            *ek_ref = survivor;
                        }
                    }
                }
                // Remove all duplicate edge references (not just consecutive ones).
                // After merging, the same EdgeKey may appear at non-consecutive positions.
                let mut seen = std::collections::HashSet::new();
                wire.edges.retain(|(ek, _)| seen.insert(*ek));
            }
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{CurveGeom, SurfaceGeom};
    use crate::geom::curve2d::Curve2d;
    use std::collections::HashMap;

    /// Build two adjacent square faces sharing one edge, leaving other edges open.
    fn make_two_adjacent_faces(reg: &mut BRepStore) -> (ShellKey, Vec<EdgeKey>) {
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO,
            normal: PVec3::Z,
            u_dir: PVec3::X,
        };

        // Face A: square (0,0)-(1,1)
        let v0 = reg.find_or_add_vertex(PVec3::new(0.0, 0.0, 0.0), 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::new(1.0, 0.0, 0.0), 1e-4);
        let v2 = reg.find_or_add_vertex(PVec3::new(1.0, 1.0, 0.0), 1e-4);
        let v3 = reg.find_or_add_vertex(PVec3::new(0.0, 1.0, 0.0), 1e-4);

        // Face B: square (1,0)-(2,1)
        let v4 = reg.find_or_add_vertex(PVec3::new(2.0, 0.0, 0.0), 1e-4);
        let v5 = reg.find_or_add_vertex(PVec3::new(2.0, 1.0, 0.0), 1e-4);

        let wka = reg.wires.insert(BRepWire { edges: vec![] });
        let fka = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wka,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        let wkb = reg.wires.insert(BRepWire { edges: vec![] });
        let fkb = reg.faces.insert(BRepFace {
            surface: surface.clone(),
            outer_wire: wkb,
            inner_wires: vec![],
            same_sense: true,
            tolerance: 1e-4,
            seam_edges: vec![],
            color: None,
            degenerated_edges: vec![],
        });

        // Face A edges
        let make_line = |a: PVec3, b: PVec3| CurveGeom::Line { origin: a, direction: b - a };
        let make_pc = |a: PVec3, b: PVec3| Curve2d::Line {
            origin: (a.x, a.y),
            direction: (b.x - a.x, b.y - a.y),
        };

        let e0 = reg.add_edge_with_pcurve(v0, v1, make_line(PVec3::new(0.,0.,0.), PVec3::new(1.,0.,0.)), 1e-4, fka,
            make_pc(PVec3::new(0.,0.,0.), PVec3::new(1.,0.,0.)), true);
        let e1 = reg.add_edge_with_pcurve(v1, v2, make_line(PVec3::new(1.,0.,0.), PVec3::new(1.,1.,0.)), 1e-4, fka,
            make_pc(PVec3::new(1.,0.,0.), PVec3::new(1.,1.,0.)), true);
        let e2 = reg.add_edge_with_pcurve(v2, v3, make_line(PVec3::new(1.,1.,0.), PVec3::new(0.,1.,0.)), 1e-4, fka,
            make_pc(PVec3::new(1.,1.,0.), PVec3::new(0.,1.,0.)), true);
        let e3 = reg.add_edge_with_pcurve(v3, v0, make_line(PVec3::new(0.,1.,0.), PVec3::new(0.,0.,0.)), 1e-4, fka,
            make_pc(PVec3::new(0.,1.,0.), PVec3::new(0.,0.,0.)), true);

        // Face B edges (e1 is shared with face A — v1→v2)
        let e4 = reg.add_edge_with_pcurve(v1, v4, make_line(PVec3::new(1.,0.,0.), PVec3::new(2.,0.,0.)), 1e-4, fkb,
            make_pc(PVec3::new(1.,0.,0.), PVec3::new(2.,0.,0.)), true);
        // Shared edge — add fkb's pcurve to e1
        if let Some(edge) = reg.edges.get_mut(e1) {
            edge.pcurves.insert(fkb, make_pc(PVec3::new(1.,0.,0.), PVec3::new(1.,1.,0.)));
        }
        reg.edge_to_faces.entry(e1).or_default().push(fkb);
        let e5 = reg.add_edge_with_pcurve(v4, v5, make_line(PVec3::new(2.,0.,0.), PVec3::new(2.,1.,0.)), 1e-4, fkb,
            make_pc(PVec3::new(2.,0.,0.), PVec3::new(2.,1.,0.)), true);
        let e6 = reg.add_edge_with_pcurve(v5, v2, make_line(PVec3::new(2.,1.,0.), PVec3::new(1.,1.,0.)), 1e-4, fkb,
            make_pc(PVec3::new(2.,1.,0.), PVec3::new(1.,1.,0.)), true);

        reg.wires.get_mut(wka).unwrap().edges = vec![
            (e0, Orientation::Forward), (e1, Orientation::Forward),
            (e2, Orientation::Forward), (e3, Orientation::Forward),
        ];
        reg.wires.get_mut(wkb).unwrap().edges = vec![
            (e1, Orientation::Forward), (e4, Orientation::Forward),
            (e5, Orientation::Forward), (e6, Orientation::Forward),
        ];

        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fka, Orientation::Forward), (fkb, Orientation::Forward)],
            closed: false,
            step_id: None,
        });

        (sk, vec![e0, e1, e2, e3, e4, e5, e6])
    }

    #[test]
    fn test_find_open_edges_adjacent_faces() {
        let mut reg = BRepStore::new();
        let (sk, _edges) = make_two_adjacent_faces(&mut reg);
        let open = find_open_edges(sk, &reg);
        // e1 is shared (2 faces), all others have 1 face → 6 open edges
        assert_eq!(open.len(), 6, "should find 6 open edges, got {}", open.len());
    }

    #[test]
    fn test_find_open_edges_no_opens() {
        // A single face has all edges open (no second face)
        let mut reg = BRepStore::new();
        let surface = SurfaceGeom::Plane {
            origin: PVec3::ZERO, normal: PVec3::Z, u_dir: PVec3::X,
        };
        let v0 = reg.find_or_add_vertex(PVec3::ZERO, 1e-4);
        let v1 = reg.find_or_add_vertex(PVec3::X, 1e-4);
        let wk = reg.wires.insert(BRepWire { edges: vec![] });
        let fk = reg.faces.insert(BRepFace {
            surface, outer_wire: wk, inner_wires: vec![],
            same_sense: true, tolerance: 1e-4, seam_edges: vec![],
            color: None, degenerated_edges: vec![],
        });
        let line = CurveGeom::Line { origin: PVec3::ZERO, direction: PVec3::X };
        let pc = Curve2d::Line { origin: (0.0, 0.0), direction: (1.0, 0.0) };
        let ek = reg.add_edge_with_pcurve(v0, v1, line, 1e-4, fk, pc, true);
        reg.wires.get_mut(wk).unwrap().edges = vec![(ek, Orientation::Forward)];
        let sk = reg.shells.insert(BRepShell {
            faces: vec![(fk, Orientation::Forward)], closed: false, step_id: None,
        });
        let open = find_open_edges(sk, &reg);
        // Single edge with 1 face → 1 open edge
        assert_eq!(open.len(), 1);
    }
}
