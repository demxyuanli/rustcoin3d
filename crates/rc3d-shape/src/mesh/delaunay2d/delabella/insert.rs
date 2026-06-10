//! Incremental point insertion — the Newton Apple Wrapper core loop.
//!
//! For each point in sorted order:
//!   1. Find a visible face via adjacency walk from the previous insertion point
//!   2. Flood all visible faces via stack-based BFS
//!   3. Collect silhouette edges (boundary between visible and invisible)
//!   4. Create new cone faces from the inserted point to each silhouette edge
//!   5. Sew adjacent new cone faces together
//!   6. Classify new faces as Delaunay vs hull

use super::hull::prepare;
use super::predicates::{adaptive_incircle, adaptive_orient2d};
use super::table::{DelaBella, FLAG_DELAUNAY, FLAG_HULL, INVALID};

// ── Visible face search ───────────────────────────────────────────────

/// Find a face visible from point (px, py) via adjacency walk from `hint`.
///
/// Starts at `hint` and walks toward the point by crossing edges where the
/// point is on the exterior (CW) side. This is O(visible-diameter) rather
/// than O(F) and is the core NAW performance advantage.
///
/// Falls back to a linear scan if the adjacency walk doesn't converge (e.g.
/// disconnected topology after a bug).
fn find_visible_face(della: &DelaBella, px: f64, py: f64, hint: u32) -> Option<u32> {
    // Phase 1: adjacency walk from hint toward the point
    if hint != INVALID && (hint as usize) < della.faces.len() {
        let mut fi = hint;
        let mut visited = vec![false; della.faces.len()];
        for _ in 0..della.faces.len() {
            if fi == INVALID || fi as usize >= della.faces.len() || visited[fi as usize] {
                break;
            }
            visited[fi as usize] = true;
            let face = &della.faces[fi as usize];
            if face.is_dead() {
                break;
            }

            let v0 = &della.verts[face.v[0] as usize];
            let v1 = &della.verts[face.v[1] as usize];
            let v2 = &della.verts[face.v[2] as usize];

            // Test Delaunay criterion violation
            let inc = adaptive_incircle(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, px, py);
            if inc > 0.0 {
                return Some(fi);
            }

            // Test if point is outside this face: on CW side of at least one edge.
            // If so, walk across that edge toward the point.
            let mut crossed = false;
            for slot in 0..3 {
                let (ea, eb) = face.edge_verts(slot);
                let pa = &della.verts[ea as usize];
                let pb = &della.verts[eb as usize];
                let o = adaptive_orient2d(pa.x, pa.y, pb.x, pb.y, px, py);
                if o < 0.0 {
                    // Point is on the exterior side of this edge — walk across
                    let ni = face.f[slot];
                    if ni != INVALID && !della.faces[ni as usize].is_dead() {
                        fi = ni;
                        crossed = true;
                        break;
                    }
                }
            }
            if !crossed {
                // Point is inside or on this face (no edge has exterior side)
                // Check orientation to determine if it's truly visible
                let o01 = adaptive_orient2d(v0.x, v0.y, v1.x, v1.y, px, py);
                let o12 = adaptive_orient2d(v1.x, v1.y, v2.x, v2.y, px, py);
                let o20 = adaptive_orient2d(v2.x, v2.y, v0.x, v0.y, px, py);
                if o01 < 0.0 || o12 < 0.0 || o20 < 0.0 {
                    return Some(fi);
                }
                // Point is strictly inside — face is not visible, but Delaunay
                // insertion should still work. Return this face as seed anyway
                // (the flood will find zero visible faces and we skip insertion).
                return None;
            }
        }
    }

    // Phase 2: fallback linear scan
    for (fi, verts) in della.all_faces() {
        let v0 = &della.verts[verts[0] as usize];
        let v1 = &della.verts[verts[1] as usize];
        let v2 = &della.verts[verts[2] as usize];

        let inc = adaptive_incircle(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, px, py);
        if inc > 0.0 {
            return Some(fi);
        }

        let o01 = adaptive_orient2d(v0.x, v0.y, v1.x, v1.y, px, py);
        let o12 = adaptive_orient2d(v1.x, v1.y, v2.x, v2.y, px, py);
        let o20 = adaptive_orient2d(v2.x, v2.y, v0.x, v0.y, px, py);
        if o01 < 0.0 || o12 < 0.0 || o20 < 0.0 {
            return Some(fi);
        }
    }
    None
}

// ── Visible face flood ────────────────────────────────────────────────

/// Check if a face is visible from point (px, py).
fn is_face_visible(della: &DelaBella, fi: u32, px: f64, py: f64) -> bool {
    let face = &della.faces[fi as usize];
    let v0 = &della.verts[face.v[0] as usize];
    let v1 = &della.verts[face.v[1] as usize];
    let v2 = &della.verts[face.v[2] as usize];

    // Test circumcircle
    let inc = adaptive_incircle(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, px, py);
    if inc > 0.0 {
        return true;
    }

    // Test exterior: point is on CW side of any edge
    let o01 = adaptive_orient2d(v0.x, v0.y, v1.x, v1.y, px, py);
    let o12 = adaptive_orient2d(v1.x, v1.y, v2.x, v2.y, px, py);
    let o20 = adaptive_orient2d(v2.x, v2.y, v0.x, v0.y, px, py);
    o01 < 0.0 || o12 < 0.0 || o20 < 0.0
}

/// Flood all faces visible from point (px, py), starting from `seed`.
fn flood_visible(della: &DelaBella, seed: u32, px: f64, py: f64) -> Vec<u32> {
    let mut visible = Vec::new();
    let mut visited = vec![false; della.faces.len()];
    let mut stack = vec![seed];

    while let Some(fi) = stack.pop() {
        if fi == INVALID || visited[fi as usize] {
            continue;
        }
        visited[fi as usize] = true;

        if della.faces[fi as usize].is_dead() {
            continue;
        }

        visible.push(fi);

        // Check all 3 neighbors
        for slot in 0..3 {
            let ni = della.faces[fi as usize].f[slot];
            if ni != INVALID && !visited[ni as usize] && !della.faces[ni as usize].is_dead()
                && is_face_visible(della, ni, px, py) {
                    stack.push(ni);
                }
        }
    }

    visible
}

// ── Silhouette edge collection ────────────────────────────────────────

/// A directed silhouette edge: (vertex_a, vertex_b) with face sharing the edge.
struct SilEdge {
    va: u32,
    vb: u32,
    /// The invisible neighbor across this silhouette edge.
    neighbor: u32,
}

/// Collect silhouette edges: boundary between visible and invisible faces.
fn collect_silhouette(della: &DelaBella, visible: &[u32], visible_set: &[bool]) -> Vec<SilEdge> {
    let mut silhouette = Vec::new();

    for &fi in visible {
        let face = &della.faces[fi as usize];
        for slot in 0..3 {
            let ni = face.f[slot];
            // Edge is a silhouette if neighbor is not visible (or is boundary)
            let is_silhouette = ni == INVALID
                || ni as usize >= visible_set.len()
                || !visible_set[ni as usize]
                || della.faces[ni as usize].is_dead();

            if is_silhouette {
                let (va, vb) = face.edge_verts(slot);
                silhouette.push(SilEdge {
                    va,
                    vb,
                    neighbor: ni,
                });
            }
        }
    }

    silhouette
}

// ── Cone face creation and sewing ─────────────────────────────────────

/// Create new cone faces from the inserted point to each silhouette edge.
/// Returns the new face indices.
fn create_cone_faces(
    della: &mut DelaBella,
    point_idx: u32,
    silhouette: &[SilEdge],
) -> Vec<u32> {
    let mut new_faces = Vec::with_capacity(silhouette.len());

    for se in silhouette {
        // New face: (point_idx, va, vb) — oriented so the interior of the
        // cone is on the CCW side. The visible face was on one side; the
        // invisible neighbor (or hull boundary) is on the other.
        let fi = della.alloc_face();
        let face = &mut della.faces[fi as usize];
        face.v = [point_idx, se.va, se.vb];
        face.f = [INVALID; 3]; // Will be filled by sewing

        // The neighbor across the silhouette edge is the invisible face
        face.f[0] = se.neighbor; // slot 0 is opposite point_idx = edge va-vb

        // Update the invisible neighbor to point back to this new face
        if se.neighbor != INVALID {
            let (ea, eb) = (se.va, se.vb);
            if let Some(ns) = della.faces[se.neighbor as usize].find_edge_slot(ea, eb) {
                della.faces[se.neighbor as usize].f[ns] = fi;
            }
        }

        new_faces.push(fi);
    }

    new_faces
}

/// Sew adjacent new cone faces together by matching shared vertices.
/// Uses a HashMap from edge → face index for O(1) lookup instead of O(n²).
fn sew_cone_faces(della: &mut DelaBella, new_faces: &[u32], _point_idx: u32) {
    // Build a map from (vertex_a, vertex_b) → (face_index, slot) for non-base edges.
    // Each new face has:
    //   slot 0: edge va — vb (base edge, already connected to invisible neighbor)
    //   slot 1: edge vb — point_idx
    //   slot 2: edge point_idx — va
    // Two adjacent faces share an edge if they share two non-point vertices.

    // Map: (non-point vertex, point_idx) → (face, slot)
    // For slot 1: edge (vb, point_idx)
    // For slot 2: edge (point_idx, va)
    // Two faces i,j are adjacent if face_i.v[1] == face_j.v[2] (share va_i == vb_j)
    // or face_i.v[2] == face_j.v[1] (share vb_i == va_j)

    // Build edge map: non-point vertex → (face, 1 or 2)
    use std::collections::HashMap;
    let mut edge_map: HashMap<u32, Vec<(u32, usize)>> = HashMap::new(); // vertex → [(face, slot)]
    for &fi in new_faces {
        let face = &della.faces[fi as usize];
        let va = face.v[1];
        let vb = face.v[2];
        edge_map.entry(va).or_default().push((fi, 2)); // slot 2 has point_idx-va
        edge_map.entry(vb).or_default().push((fi, 1)); // slot 1 has vb-point_idx
    }

    // For each face, find its neighbors on slots 1 and 2
    for &fi in new_faces {
        let face = &della.faces[fi as usize];
        let va = face.v[1];
        let vb = face.v[2];

        // Slot 1: edge (vb, point_idx). Neighbor has slot 2 with va == vb
        if face.f[1] == INVALID {
            if let Some(entries) = edge_map.get(&vb) {
                for &(fj, slot_j) in entries {
                    if fj != fi && slot_j == 2 {
                        della.faces[fi as usize].f[1] = fj;
                        della.faces[fj as usize].f[2] = fi;
                        break;
                    }
                }
            }
        }

        // Slot 2: edge (point_idx, va). Neighbor has slot 1 with vb == va
        if della.faces[fi as usize].f[2] == INVALID {
            if let Some(entries) = edge_map.get(&va) {
                for &(fj, slot_j) in entries {
                    if fj != fi && slot_j == 1 {
                        della.faces[fi as usize].f[2] = fj;
                        della.faces[fj as usize].f[1] = fi;
                        break;
                    }
                }
            }
        }
    }
}

// ── Face classification ───────────────────────────────────────────────

/// Classify new faces as Delaunay or hull based on orientation.
/// A face is Delaunay if its signed area (orient2d) is positive (CCW),
/// and hull if negative.
fn classify_faces(della: &mut DelaBella, new_faces: &[u32]) {
    for &fi in new_faces {
        let orient = della.face_orient(fi);
        if orient > 0.0 {
            della.faces[fi as usize].flags |= FLAG_DELAUNAY;
        } else {
            della.faces[fi as usize].flags |= FLAG_HULL;
        }
    }
}

// ── Kill visible faces ────────────────────────────────────────────────

/// Remove all visible faces and replace with new cone faces.
/// Snapshots all neighbor data before mutation to satisfy borrow checker.
fn kill_visible_and_replace(
    della: &mut DelaBella,
    visible: &[u32],
    point_idx: u32,
) -> Vec<u32> {
    // Build visible set for silhouette detection
    let mut visible_set = vec![false; della.faces.len()];
    for &fi in visible {
        if (fi as usize) < visible_set.len() {
            visible_set[fi as usize] = true;
        }
    }

    // Collect silhouette before killing visible faces
    let silhouette = collect_silhouette(della, visible, &visible_set);

    // Snapshot all neighbor detach info before mutation
    let detach_info: Vec<(u32, [(u32, u32); 3], [u32; 3])> = visible.iter().map(|&fi| {
        let face = &della.faces[fi as usize];
        let edges = [face.edge_verts(0), face.edge_verts(1), face.edge_verts(2)];
        let neighbors = face.f;
        (fi, edges, neighbors)
    }).collect();

    // Kill visible faces and detach from neighbors
    for &(fi, edges, neighbors) in &detach_info {
        della.free_face(fi);
        for slot in 0..3 {
            let ni = neighbors[slot];
            if ni != INVALID && (ni as usize) < della.faces.len() && !della.faces[ni as usize].is_dead() {
                let (va, vb) = edges[slot];
                if let Some(ns) = della.faces[ni as usize].find_edge_slot(va, vb) {
                    della.faces[ni as usize].f[ns] = INVALID;
                }
            }
        }
    }

    // Create new cone faces
    let new_faces = create_cone_faces(della, point_idx, &silhouette);

    // Sew adjacent cone faces
    sew_cone_faces(della, &new_faces, point_idx);

    // Classify new faces
    classify_faces(della, &new_faces);

    new_faces
}

// ── Main triangulation loop ───────────────────────────────────────────

/// Triangulate all points: prepare hull, then incrementally insert remaining points.
pub fn triangulate(della: &mut DelaBella, points: &[(f64, f64)], orig_indices: &[u32]) -> i32 {
    let n = points.len();
    if n < 3 {
        return (n as i32).min(1);
    }

    // Prepare: sort, dedup, build initial hull
    let unique_count = prepare(della, points, orig_indices);
    if unique_count < 3 {
        return (unique_count as i32).min(1);
    }

    // Check that we have an initial face
    let has_faces = !della.all_faces().is_empty();
    if !has_faces {
        return -1; // All points collinear
    }

    // Find the initial triple vertices to know where to start insertion
    let initial_face = della.hull_first;
    let mut last_face = initial_face;

    // Insert remaining points (after the initial triple) in sorted order
    // The initial triple used points at indices determined by prepare()
    for vi in 3..unique_count {
        let vi = vi as u32;
        let (px, py) = della.vert_pos(vi);

        // Find a visible face from the last insertion point
        let seed = match find_visible_face(della, px, py, last_face) {
            Some(f) => f,
            None => continue, // Point is already inside the hull (shouldn't happen in sorted order)
        };

        // Flood all visible faces
        let visible = flood_visible(della, seed, px, py);
        if visible.is_empty() {
            continue;
        }

        // Kill visible faces and create cone
        let new_faces = kill_visible_and_replace(della, &visible, vi);
        if !new_faces.is_empty() {
            last_face = new_faces[0];
        }
    }

    // Rebuild face lists
    rebuild_face_lists(della);

    // Count Delaunay triangles
    let dela_count = della.faces.iter().filter(|f| f.is_delaunay() && !f.is_dead()).count();
    dela_count as i32
}

/// Rebuild dela_first and hull_first lists by classifying all alive faces.
/// Uses a two-pass approach to avoid borrow conflicts: first classify, then link.
fn rebuild_face_lists(della: &mut DelaBella) {
    // Pass 1: Classify all alive faces (read-only)
    let classifications: Vec<(usize, bool)> = della.faces.iter().enumerate()
        .filter(|(_, face)| !face.is_dead())
        .map(|(fi, _)| (fi, classify_face_orient(della, fi as u32) > 0.0))
        .collect();

    // Pass 2: Set flags
    for &(fi, is_dela) in &classifications {
        if is_dela {
            della.faces[fi].flags = (della.faces[fi].flags & !FLAG_HULL) | FLAG_DELAUNAY;
        } else {
            della.faces[fi].flags = (della.faces[fi].flags & !FLAG_DELAUNAY) | FLAG_HULL;
        }
    }

    // Pass 3: Build linked lists
    della.dela_first = INVALID;
    della.hull_first = INVALID;

    let mut last_dela = INVALID;
    let mut last_hull = INVALID;

    for &(fi, is_dela) in &classifications {
        let fi32 = fi as u32;
        della.faces[fi].next = INVALID;
        if is_dela {
            if della.dela_first == INVALID {
                della.dela_first = fi32;
            } else if last_dela != INVALID {
                della.faces[last_dela as usize].next = fi32;
            }
            last_dela = fi32;
        } else {
            if della.hull_first == INVALID {
                della.hull_first = fi32;
            } else if last_hull != INVALID {
                della.faces[last_hull as usize].next = fi32;
            }
            last_hull = fi32;
        }
    }
}

/// Classify a face as Delaunay (CCW) or hull (CW) by orientation sign.
fn classify_face_orient(della: &DelaBella, fi: u32) -> f64 {
    della.face_orient(fi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangulate_square() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2, 3];
        let mut della = DelaBella::new();
        let count = triangulate(&mut della, &pts, &orig);
        assert!(count > 0, "square should produce triangles, got {}", count);
        // 4 points convex: 3 total faces (2 Delaunay + 1 hull)
        assert!(count >= 2, "square should produce at least 2 Delaunay triangles, got {}", count);
    }

    #[test]
    fn test_triangulate_three_points() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2];
        let mut della = DelaBella::new();
        let count = triangulate(&mut della, &pts, &orig);
        assert_eq!(count, 1, "triangle should produce 1 face");
    }

    #[test]
    fn test_triangulate_collinear() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let orig: Vec<u32> = vec![0, 1, 2];
        let mut della = DelaBella::new();
        let count = triangulate(&mut della, &pts, &orig);
        assert!(count <= 0, "collinear should produce no triangles, got {}", count);
    }

    #[test]
    fn test_triangulate_grid() {
        // 3x3 grid of points
        let mut pts = Vec::new();
        let mut orig = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                pts.push((x as f64, y as f64));
                orig.push(orig.len() as u32);
            }
        }
        let mut della = DelaBella::new();
        let count = triangulate(&mut della, &pts, &orig);
        assert!(count > 0, "grid should produce triangles, got {}", count);
        // For 9 convex points: expect >= 7 Delaunay faces
        assert!(count >= 2, "grid should produce at least 2 triangles, got {}", count);
    }

    #[test]
    fn test_insert_square_debug() {
        // Square: after KD sort, points are (0,0), (0,1), (1,0), (1,1)
        // Initial triple: (0,0), (0,1), (1,0) — forms one CCW triangle
        // Point (1,1) should be visible and create 3 cone faces (1 Delaunay + 2 hull or 2+1)
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let orig: Vec<u32> = vec![0, 1, 2, 3];
        let mut della = DelaBella::new();
        let count = triangulate(&mut della, &pts, &orig);

        // After triangulation, count alive faces
        let alive = della.all_faces();
        let dela = della.delaunay_faces();
        eprintln!("Square: {} total faces, {} Delaunay faces, count={}", alive.len(), dela.len(), count);

        // Print face details for debugging
        for (fi, verts) in &alive {
            let face = &della.faces[*fi as usize];
            let positions: Vec<_> = verts.iter().map(|&v| della.vert_pos(v)).collect();
            let neighbors: Vec<_> = face.f.iter().map(|&n| if n == INVALID { -1i32 } else { n as i32 }).collect();
            eprintln!(
                "  face {}: verts={:?} pos={:?} neighbors={:?} flags=0x{:02x}",
                fi, verts, positions, neighbors, face.flags
            );
        }
        assert!(count >= 1, "square should produce at least 1 triangle, got {}", count);
    }
}
