//! Constrained Delaunay triangulation: edge constraint enforcement.
//!
//! Implements the DelaBella constraint algorithm:
//!   1. Find edges crossing the constraint via orient2d walk
//!   2. Flip crossing edges in a loop (with concavity check)
//!   3. Restore Delaunay property via Lawson edge-flip passes
//!
//! The edge-bit system tracks constraint ("fixed") status through flips.

use super::predicates::adaptive_orient2d;
use super::table::{DelaBella, INVALID};

// ── Find face containing an edge endpoint ─────────────────────────────

/// Find a face that has vertex `vi` and is adjacent to a face containing
/// the start of a walk toward `vj`.
fn find_face_with_vert(della: &DelaBella, vi: u32) -> Option<u32> {
    for (fi, verts) in della.all_faces() {
        if verts.contains(&vi) {
            return Some(fi);
        }
    }
    None
}

/// Check if an edge (va, vb) already exists in the mesh as a face edge.
fn edge_exists(della: &DelaBella, va: u32, vb: u32) -> Option<(u32, usize)> {
    for (fi, _) in della.all_faces() {
        let face = &della.faces[fi as usize];
        if let Some(slot) = face.find_edge_slot(va, vb) {
            return Some((fi, slot));
        }
    }
    None
}

// ── Crossing edge detection ───────────────────────────────────────────

/// Check if segment (ax,ay)→(bx,by) properly crosses the edge at `slot` of face `fi`.
fn segments_cross_proper(
    della: &DelaBella,
    fi: u32,
    slot: usize,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
) -> bool {
    let face = &della.faces[fi as usize];
    let (ea, eb) = face.edge_verts(slot);
    let pa = della.vert_pos(ea);
    let pb = della.vert_pos(eb);

    // Proper intersection: both endpoints of each segment are on opposite sides of the other
    let o1 = adaptive_orient2d(ax, ay, bx, by, pa.0, pa.1);
    let o2 = adaptive_orient2d(ax, ay, bx, by, pb.0, pb.1);
    let o3 = adaptive_orient2d(pa.0, pa.1, pb.0, pb.1, ax, ay);
    let o4 = adaptive_orient2d(pa.0, pa.1, pb.0, pb.1, bx, by);

    // Proper crossing: signs are opposite for both pairs
    (o1 > 0.0 && o2 < 0.0 || o1 < 0.0 && o2 > 0.0)
        && (o3 > 0.0 && o4 < 0.0 || o3 < 0.0 && o4 > 0.0)
}

// ── Edge flip ─────────────────────────────────────────────────────────

/// Flip the edge shared between face `fi` at `slot` and its neighbor.
/// The quad formed by the two triangles has its diagonal swapped.
///
/// Topology before flip (shared edge ea–eb):
/// ```text
///        va
///       / \
///     eb   ea    ← shared edge
///       \ /
///        vb
/// ```
///
/// External neighbors:
/// - `fi_nea` shares edge va–eb with fi (opposite ea)
/// - `fi_neb` shares edge va–ea with fi (opposite eb)
/// - `ni_nea` shares edge vb–eb with ni (opposite ea)
/// - `ni_neb` shares edge vb–ea with ni (opposite eb)
///
/// After flip (new diagonal va–vb):
/// - new fi (va, ea, vb): edges ea–vb, va–vb (shared), va–ea
/// - new ni (va, vb, eb): edges vb–eb, va–eb, va–vb (shared)
///
/// Edge migration:
/// - va–eb moves from fi → ni  →  fi_nea must point to ni
/// - vb–ea moves from ni → fi  →  ni_neb must point to fi
/// - va–ea stays in fi  →  fi_neb still points to fi (unchanged)
/// - vb–eb stays in ni  →  ni_nea still points to ni (unchanged)
///
/// Returns true if the flip was performed, false if the quad is concave.
fn flip_edge(della: &mut DelaBella, fi: u32, slot: usize) -> bool {
    let face = &della.faces[fi as usize];
    let ni = face.f[slot];
    if ni == INVALID {
        return false;
    }

    let va = face.v[slot]; // opposite vertex in face fi
    let (ea, eb) = face.edge_verts(slot); // the shared edge

    // Find the opposite vertex in neighbor
    let nface = &della.faces[ni as usize];
    let nslot = match nface.find_edge_slot(ea, eb) {
        Some(s) => s,
        None => return false,
    };
    let vb = nface.v[nslot]; // opposite vertex in neighbor

    // Check convexity: the quad (va, ea, vb, eb) must be convex
    let pa = della.vert_pos(va);
    let pb = della.vert_pos(vb);
    let pe_a = della.vert_pos(ea);
    let pe_b = della.vert_pos(eb);

    let o1 = adaptive_orient2d(pa.0, pa.1, pe_a.0, pe_a.1, pb.0, pb.1);
    let o2 = adaptive_orient2d(pa.0, pa.1, pb.0, pb.1, pe_b.0, pe_b.1);
    if o1 * o2 <= 0.0 {
        return false; // Concave quad — can't flip
    }

    // Collect external neighbor info before restructuring
    // fi_nea = neighbor sharing edge va-eb (opposite vertex ea in old fi)
    // fi_neb = neighbor sharing edge va-ea (opposite vertex eb in old fi)
    // ni_nea = neighbor sharing edge vb-eb (opposite vertex ea in old ni)
    // ni_neb = neighbor sharing edge vb-ea (opposite vertex eb in old ni)
    let (fi_nea, fi_neb, ni_nea, ni_neb) = {
        let fi_face = &della.faces[fi as usize];
        let ni_face = &della.faces[ni as usize];
        let fi_sea = fi_face.find_vert(ea).unwrap();
        let fi_seb = fi_face.find_vert(eb).unwrap();
        let ni_sea = ni_face.find_vert(ea).unwrap();
        let ni_seb = ni_face.find_vert(eb).unwrap();
        (
            fi_face.f[fi_sea], // neighbor sharing edge va-eb
            fi_face.f[fi_seb], // neighbor sharing edge va-ea
            ni_face.f[ni_sea], // neighbor sharing edge vb-eb
            ni_face.f[ni_seb], // neighbor sharing edge vb-ea
        )
    };

    // Save edge fixed bits before restructuring
    let edge_fixed_fi_slot = della.faces[fi as usize].is_edge_fixed(slot);
    let edge_fixed_ni_slot = della.faces[ni as usize].is_edge_fixed(nslot);

    // --- Restructure ---
    // new fi: (va, ea, vb)
    //   slot 0 opposite va → edge ea-vb → ni_neb (edge vb-ea moved from ni to fi)
    //   slot 1 opposite ea → edge va-vb → ni (shared diagonal)
    //   slot 2 opposite vb → edge va-ea → fi_neb (stays in fi)
    {
        let face = &mut della.faces[fi as usize];
        face.v = [va, ea, vb];
        face.f = [ni_neb, ni, fi_neb];
    }

    // new ni: (va, vb, eb)
    //   slot 0 opposite va → edge vb-eb → ni_nea (stays in ni)
    //   slot 1 opposite vb → edge va-eb → fi_nea (edge va-eb moved from fi to ni)
    //   slot 2 opposite eb → edge va-vb → fi (shared diagonal)
    {
        let nface = &mut della.faces[ni as usize];
        nface.v = [va, vb, eb];
        nface.f = [ni_nea, fi_nea, fi];
    }

    // --- Update back-references for edges that migrated between faces ---
    // Edge va-eb moved from fi → ni: fi_nea must now point to ni instead of fi
    if fi_nea != INVALID {
        redirect_neighbor(della, fi_nea, fi, ni);
    }
    // Edge vb-ea moved from ni → fi: ni_neb must now point to fi instead of ni
    if ni_neb != INVALID {
        redirect_neighbor(della, ni_neb, ni, fi);
    }

    // Transfer edge bits: the old diagonal fixed bits move to new positions
    della.faces[fi as usize].set_edge_fixed(1, edge_fixed_fi_slot);
    della.faces[ni as usize].set_edge_fixed(2, edge_fixed_ni_slot);

    true
}

/// Update a neighbor's f[] so that the slot which used to point to `old_face`
/// now points to `new_face`. Required when an edge migrates from one face to
/// the other during a flip.
fn redirect_neighbor(della: &mut DelaBella, neighbor_fi: u32, old_face: u32, new_face: u32) {
    let neighbor = &mut della.faces[neighbor_fi as usize];
    for slot in 0..3 {
        if neighbor.f[slot] == old_face {
            neighbor.f[slot] = new_face;
            return;
        }
    }
}

// ── Constraint enforcement ────────────────────────────────────────────

/// Enforce a constraint edge between vertices va and vb.
/// Returns true if the constraint was successfully enforced.
pub fn constrain_edge(della: &mut DelaBella, va: u32, vb: u32) -> bool {
    let (pa, pb) = (della.vert_pos(va), della.vert_pos(vb));

    // Quick check: edge already exists
    if let Some((fi, slot)) = edge_exists(della, va, vb) {
        della.faces[fi as usize].mark_edge_fixed(slot);
        // Also mark in the neighbor
        let ni = della.faces[fi as usize].f[slot];
        if ni != INVALID {
            if let Some(ns) = della.faces[ni as usize].find_edge_slot(va, vb) {
                della.faces[ni as usize].mark_edge_fixed(ns);
            }
        }
        return true;
    }

    // Find edges that cross the constraint segment
    let mut offending: Vec<(u32, usize)> = Vec::new();
    for (fi, _) in della.all_faces() {
        for slot in 0..3 {
            if segments_cross_proper(della, fi, slot, pa.0, pa.1, pb.0, pb.1) {
                offending.push((fi, slot));
            }
        }
    }

    // Flip loop: iteratively flip crossing edges
    let mut max_iter = 128;
    let mut flipped: Vec<u32> = Vec::new();

    while !offending.is_empty() && max_iter > 0 {
        max_iter -= 1;
        let (fi, slot) = offending.remove(0);
        if della.faces[fi as usize].is_dead() {
            continue;
        }

        if flip_edge(della, fi, slot) {
            flipped.push(fi);

            // Check if constraint now exists
            if let Some((efi, eslot)) = edge_exists(della, va, vb) {
                della.faces[efi as usize].mark_edge_fixed(eslot);
                let eni = della.faces[efi as usize].f[eslot];
                if eni != INVALID {
                    if let Some(ens) = della.faces[eni as usize].find_edge_slot(va, vb) {
                        della.faces[eni as usize].mark_edge_fixed(ens);
                    }
                }
                // Restore Delaunay property around flipped edges
                lawson_restore(della, &flipped);
                return true;
            }

            // Check edges of the two flipped faces for new crossings.
            // Only fi and its neighbor changed — other faces are unaffected.
            let ni = della.faces[fi as usize].f[1]; // slot 1 = new shared diagonal
            for check_fi in [fi, ni] {
                if check_fi != INVALID && !della.faces[check_fi as usize].is_dead() {
                    for s in 0..3 {
                        if segments_cross_proper(della, check_fi, s, pa.0, pa.1, pb.0, pb.1) {
                            offending.push((check_fi, s));
                        }
                    }
                }
            }
        }
    }

    // If flip loop didn't work, try once more with a fresh scan
    if !offending.is_empty() {
        return false;
    }

    // Mark as constraint if it exists now
    if let Some((efi, eslot)) = edge_exists(della, va, vb) {
        della.faces[efi as usize].mark_edge_fixed(eslot);
        return true;
    }

    false
}

// ── Lawson edge-flip restoration ──────────────────────────────────────

/// Restore Delaunay property by flipping non-Delaunay edges.
fn lawson_restore(della: &mut DelaBella, affected: &[u32]) {
    let max_passes = 4;
    for _ in 0..max_passes {
        let mut flipped = 0;
        // Collect candidate edges
        let mut candidates: Vec<(u32, usize)> = Vec::new();
        for &fi in affected {
            if della.faces[fi as usize].is_dead() {
                continue;
            }
            for slot in 0..3 {
                if della.faces[fi as usize].is_edge_fixed(slot) {
                    continue; // Don't flip constrained edges
                }
                candidates.push((fi, slot));
            }
        }
        // Also check neighbors of affected faces
        for &fi in affected {
            if della.faces[fi as usize].is_dead() {
                continue;
            }
            for slot in 0..3 {
                let ni = della.faces[fi as usize].f[slot];
                if ni != INVALID && !della.faces[ni as usize].is_dead() {
                    for ns in 0..3 {
                        if !della.faces[ni as usize].is_edge_fixed(ns) {
                            candidates.push((ni, ns));
                        }
                    }
                }
            }
        }

        for &(fi, slot) in &candidates {
            if della.faces[fi as usize].is_dead() {
                continue;
            }
            let ni = della.faces[fi as usize].f[slot];
            if ni == INVALID || della.faces[ni as usize].is_dead() {
                continue;
            }

            // Check Delaunay criterion: flip if the opposite vertex of one triangle
            // is inside the circumcircle of the other
            let face = &della.faces[fi as usize];

            let nface = &della.faces[ni as usize];
            let nslot = match nface.find_edge_slot(
                face.edge_verts(slot).0,
                face.edge_verts(slot).1,
            ) {
                Some(s) => s,
                None => continue,
            };
            let opp_ni = nface.v[nslot];

            // Test: is opp_ni inside circumcircle of fi?
            let (p0, p1, p2) = {
                let v0 = della.vert_pos(face.v[0]);
                let v1 = della.vert_pos(face.v[1]);
                let v2 = della.vert_pos(face.v[2]);
                (v0, v1, v2)
            };
            let p_opp = della.vert_pos(opp_ni);

            let inc = super::predicates::adaptive_incircle(
                p0.0, p0.1, p1.0, p1.1, p2.0, p2.1, p_opp.0, p_opp.1,
            );

            if inc > 0.0 {
                if flip_edge(della, fi, slot) {
                    flipped += 1;
                }
            }
        }

        if flipped == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::delaunay2d::delabella::insert::triangulate;
    use crate::mesh::delaunay2d::delabella::table::DelaBella;

    #[test]
    fn test_constrain_existing_edge() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2];
        let mut della = DelaBella::new();
        triangulate(&mut della, &pts, &orig);
        // After triangulation, look for any edge that exists and constrain it
        let found = edge_exists(&della, 0, 1)
            .or_else(|| edge_exists(&della, 0, 2))
            .or_else(|| edge_exists(&della, 1, 2));
        if let Some((fi, slot)) = found {
            // Snapshot vertex indices before mutable borrow
            let va = della.faces[fi as usize].v[(slot + 1) % 3];
            let vb = della.faces[fi as usize].v[(slot + 2) % 3];
            let result = constrain_edge(&mut della, va, vb);
            assert!(result, "should constrain existing edge");
        }
    }

    #[test]
    fn test_constrain_diagonal_of_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2, 3];
        let mut della = DelaBella::new();
        let n = triangulate(&mut della, &pts, &orig);
        assert!(n > 0);

        // Constrain diagonal (0, 2) — may or may not exist already
        let result = constrain_edge(&mut della, 0, 2);
        assert!(result, "should be able to constrain diagonal");
    }
}
